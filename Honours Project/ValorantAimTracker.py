from ultralyticsplus import YOLO, render_result
import torch
import ultralytics.nn.tasks
import ultralytics.nn.modules
import cv2
import os
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import queue
import threading
import time
import multiprocessing as mp
from multiprocessing import Pool

# Monkeypatch torch.load so that, when weights_only is not provided,
# it defaults to False (changes behavior at runtime only; does not modify torch library files).
if not hasattr(torch.load, '_is_patched'):
    _original_torch_load = torch.load
    def _load_with_weights_only_default_false(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    _load_with_weights_only_default_false._is_patched = True
    torch.load = _load_with_weights_only_default_false

# load model
model = YOLO('keremberke/yolov8m-valorant-detection')

# set model parameters
model.overrides['conf'] = 0.8  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image


def _process_frame(args):
    """Worker function for multiprocessing pool that detects enemies in a frame."""
    frame_number, frame = args
    try:
        results = model.predict(frame)
        enemy_detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                if 'enemy' in class_name.lower() and confidence >= 0.8:
                    enemy_detections.append({
                        'frame': frame_number,
                        'class': class_name,
                        'confidence': confidence,
                        'box': box.xyxy[0].cpu().numpy()
                    })
        
        if enemy_detections:
            return {
                'frame_number': frame_number,
                'frame_image': frame.copy(),
                'detections': enemy_detections,
                'results': results[0]
            }
        return None
    except Exception as e:
        print(f"Error processing frame {frame_number}: {e}")
        return None


def detect_enemies_in_video(video_path, max_detected_frames_to_display=10, num_workers=4):
    """
    Process video frame by frame to detect enemies using multiprocessing.
    
    Args:
        video_path (str): Path to the video file
        max_detected_frames_to_display (int): Maximum number of detected frames to display
        num_workers (int): Number of worker processes for parallel processing
    """
    
    start_time = time.time()
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = 3  # Process every 3th frame
    
    print(f"Video loaded: {total_frames} total frames at {fps} FPS")
    print(f"Processing every {frame_skip}th frame for enemies...")
    print(f"Using {num_workers} worker processes for parallel processing\n")
    
    # Collect frames to process
    frames_to_process = []
    frame_count = 0
    frames_queued = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            frames_to_process.append((frame_count, frame))
            frames_queued += 1
            
            if frames_queued % 100 == 0:
                print(f"Queued {frames_queued} frames for processing...")
        
        frame_count += 1
    
    cap.release()
    print(f"Queued {frames_queued} frames total for processing.\n")
    
    # Process frames in parallel using multiprocessing Pool
    detected_frames = []
    
    if frames_to_process:
        # Use spawn context for more predictable process creation and model initialization
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_workers) as pool:
            # Higher chunksize reduces task scheduling overhead
            results = pool.imap_unordered(_process_frame, frames_to_process, chunksize=8)
            
            processed_count = 0
            for result in results:
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count}/{frames_queued} frames...")
                
                if result is not None:
                    detected_frames.append(result)
    
    # Sort by frame number to maintain order
    detected_frames.sort(key=lambda x: x['frame_number'])
    
    print(f"\nDone! Total detected frames with enemies: {len(detected_frames)}\n")
    
    # Calculate crosshair position for all detected frames
    if detected_frames:
        detected_frames = calculate_crosshair_position(detected_frames)
    
    # Save detected frames to detections folder
    if detected_frames:
        detections_folder = os.path.join(os.path.dirname(video_path), 'detections')
        os.makedirs(detections_folder, exist_ok=True)
        
        frames_to_save = min(max_detected_frames_to_display, len(detected_frames))
        print(f"Saving {frames_to_save} detected frames to '{detections_folder}':")
        
        for i, detection_info in enumerate(detected_frames[:frames_to_save]):
            print(f"\n--- Detection {i+1}/{frames_to_save} ---")
            print(f"Frame: {detection_info['frame_number']}")
            print(f"Crosshair Position: {detection_info['crosshair_position']}")
            for detection in detection_info['detections']:
                print(f"  {detection['class']}: {detection['confidence']:.2f} confidence")
            
            render = render_result(model=model, image=detection_info['frame_image'], result=detection_info['results'])
            
            if isinstance(render, Image.Image):
                render_array = np.array(render)
                render_array = cv2.cvtColor(render_array, cv2.COLOR_RGB2BGR)
            else:
                render_array = render
            
            output_path = os.path.join(detections_folder, f"detection_{i+1}_frame_{detection_info['frame_number']}.jpg")
            cv2.imwrite(output_path, render_array)
            print(f"  Saved to: {output_path}")
    
    elapsed_time = time.time() - start_time
    return elapsed_time


def detect_enemies_in_video_single_threaded(video_path, max_detected_frames_to_display=10):
    """
    Process video frame by frame to detect enemies (single-threaded).
    
    Args:
        video_path (str): Path to the video file
        max_detected_frames_to_display (int): Maximum number of detected frames to display
    """
    
    start_time = time.time()
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frame_skip = 5  # Process every 5th frame
    detected_frames = []  # Store information about frames with detections
    
    print(f"Video loaded: {total_frames} total frames at {fps} FPS")
    print(f"Processing every {frame_skip}th frame for enemies...\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Only process every 5th frame
        if frame_count % frame_skip == 0:
            # Perform inference on this frame
            results = model.predict(frame)
            
            # Filter results for enemies only
            enemy_detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # Filter for enemy class with confidence threshold of 0.8
                    confidence = float(box.conf[0])
                    if 'enemy' in class_name.lower() and confidence >= 0.8:
                        enemy_detections.append({
                            'frame': frame_count,
                            'class': class_name,
                            'confidence': confidence,
                            'box': box.xyxy[0].cpu().numpy()
                        })
            
            # Print detections for this frame if enemies found
            if enemy_detections:
                print(f"Frame {frame_count}: Found {len(enemy_detections)} enemy/enemies")
                for detection in enemy_detections:
                    print(f"  - {detection['class']}: {detection['confidence']:.2f} confidence")
                
                # Store frame info for later processing
                detected_frames.append({
                    'frame_number': frame_count,
                    'frame_image': frame.copy(),
                    'detections': enemy_detections,
                    'results': results[0]
                })
        
        frame_count += 1
        
        # Optional: Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed frame {frame_count}/{total_frames}")
    
    cap.release()
    print(f"\nDone! Processed {frame_count} frames total, checked {frame_count // frame_skip} frames.")
    print(f"Total detected frames with enemies: {len(detected_frames)}\n")
    
    # Calculate crosshair position for all detected frames
    if detected_frames:
        detected_frames = calculate_crosshair_position(detected_frames)
    
    # Save detected frames to detections folder
    if detected_frames:
        # Create detections folder if it doesn't exist
        detections_folder = os.path.join(os.path.dirname(video_path), 'detections')
        os.makedirs(detections_folder, exist_ok=True)
        
        frames_to_save = min(max_detected_frames_to_display, len(detected_frames))
        print(f"Saving {frames_to_save} detected frames to '{detections_folder}':")
        
        for i, detection_info in enumerate(detected_frames[:frames_to_save]):
            print(f"\n--- Detection {i+1}/{frames_to_save} ---")
            print(f"Frame: {detection_info['frame_number']}")
            print(f"Crosshair Position: {detection_info['crosshair_position']}")
            for detection in detection_info['detections']:
                print(f"  {detection['class']}: {detection['confidence']:.2f} confidence")
            
            # Render the frame with annotations
            render = render_result(model=model, image=detection_info['frame_image'], result=detection_info['results'])
            
            # Convert render result to numpy array if it's a PIL Image
            if isinstance(render, Image.Image):
                render_array = np.array(render)
                # Convert RGB to BGR for cv2.imwrite
                render_array = cv2.cvtColor(render_array, cv2.COLOR_RGB2BGR)
            else:
                # Already in BGR format
                render_array = render
            
            # Save the frame
            output_path = os.path.join(detections_folder, f"detection_{i+1}_frame_{detection_info['frame_number']}.jpg")
            cv2.imwrite(output_path, render_array)
            print(f"  Saved to: {output_path}")
    
    elapsed_time = time.time() - start_time
    return elapsed_time

def calculate_crosshair_position(frame_or_frames):
    """
    Calculate the centre pixel position (crosshair) for detected frames.
    
    Args:
        frame_or_frames: Either a single frame (numpy array) or a list of detection dictionaries
    
    Returns:
        If input is a single frame: tuple (x, y) representing centre coordinates
        If input is a list of detections: list of detection dictionaries with added 'crosshair_position' key
    """
    
    # Case 1: Single frame (numpy array)
    if isinstance(frame_or_frames, np.ndarray):
        height, width = frame_or_frames.shape[:2]
        crosshair_x = width // 2
        crosshair_y = height // 2
        return (crosshair_x, crosshair_y)
    
    # Case 2: List of detection dictionaries
    elif isinstance(frame_or_frames, list):
        for detection_info in frame_or_frames:
            frame = detection_info['frame_image']
            height, width = frame.shape[:2]
            crosshair_x = width // 2
            crosshair_y = height // 2
            detection_info['crosshair_position'] = (crosshair_x, crosshair_y)
        
        return frame_or_frames
    
    else:
        raise TypeError("Input must be a numpy array (single frame) or list of detection dictionaries")


def display_times(single_thread_time, multi_thread_time):
    """Display both execution times and state which approach was faster."""
    print('\n' + '='*60)
    print(f"Single-threaded Detection - Total Time: {single_thread_time:.2f} seconds")
    print(f"Multithreaded Detection - Total Time: {multi_thread_time:.2f} seconds")
    if multi_thread_time < single_thread_time:
        print(f"Multithreaded was faster by {single_thread_time - multi_thread_time:.2f} seconds")
    elif multi_thread_time > single_thread_time:
        print(f"Single-threaded was faster by {multi_thread_time - single_thread_time:.2f} seconds")
    else:
        print("Both methods took the same time")
    print('='*60 + '\n')


# Run the detection on a video file
import os
video_path = os.path.join(os.path.dirname(__file__), 'testVod.mp4')

if __name__ == '__main__':
    # Check number of CPU cores
    num_cores = mp.cpu_count()
    print(f"System has {num_cores} CPU cores\n")
    
    if num_cores < 4:
        print("Running single-threaded detection (system has fewer than 4 cores)...\n")
        elapsed = detect_enemies_in_video_single_threaded(video_path, max_detected_frames_to_display=5)
        print(f"\nTotal Time: {elapsed:.2f} seconds")
    else:
        print(f"Running multiprocessing detection (system has {num_cores} cores)...\n")
        elapsed = detect_enemies_in_video(video_path, max_detected_frames_to_display=25, num_workers=4)
        print(f"\nTotal Time: {elapsed:.2f} seconds")