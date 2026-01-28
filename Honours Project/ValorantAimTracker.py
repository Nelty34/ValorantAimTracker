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
model.overrides['conf'] = 0.7  # NMS confidence threshold (updated to 0.7)
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image


def detect_enemies_in_video(video_path, max_detected_frames_to_display=10, num_workers=4):
    """
    Process video frame by frame to detect enemies using multithreading.
    
    Args:
        video_path (str): Path to the video file
        max_detected_frames_to_display (int): Maximum number of detected frames to display
        num_workers (int): Number of worker threads for parallel processing
    """
    
    start_time = time.time()
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = 5  # Process every 5th frame
    
    # Queues for producer-consumer pattern
    frame_queue = queue.Queue(maxsize=num_workers * 2)
    result_queue = queue.Queue()
    
    print(f"Video loaded: {total_frames} total frames at {fps} FPS")
    print(f"Processing every {frame_skip}th frame for enemies...")
    print(f"Using {num_workers} worker threads for parallel processing\n")
    
    stop_event = threading.Event()
    
    def worker():
        """Worker thread that processes frames from the queue"""
        while not stop_event.is_set():
            try:
                frame_data = frame_queue.get(timeout=1)
                if frame_data is None:  # Sentinel value to stop worker
                    break
                
                frame_number, frame = frame_data
                
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
                                'frame': frame_number,
                                'class': class_name,
                                'confidence': confidence,
                                'box': box.xyxy[0].cpu().numpy()
                            })
                
                # Put result in result queue if enemies found
                if enemy_detections:
                    result_queue.put({
                        'frame_number': frame_number,
                        'frame_image': frame.copy(),
                        'detections': enemy_detections,
                        'results': results[0]
                    })
                
                frame_queue.task_done()
            except queue.Empty:
                continue
    
    # Start worker threads
    workers = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        workers.append(t)
    
    frames_queued = 0
    
    def producer():
        """Producer thread that reads frames from video"""
        nonlocal frames_queued
        frame_count = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Only process every 5th frame
            if frame_count % frame_skip == 0:
                frame_queue.put((frame_count, frame))
                frames_queued += 1
                
                if frames_queued % 100 == 0:
                    print(f"Queued {frames_queued} frames for processing...")
            
            frame_count += 1
        
        # Signal workers to stop by putting sentinel values
        for _ in range(num_workers):
            frame_queue.put(None)
    
    # Start producer thread
    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
    
    # Wait for all frames to be queued
    producer_thread.join()
    
    # Wait for all frames to be processed
    frame_queue.join()
    
    cap.release()
    
    # Signal workers to stop
    stop_event.set()
    for worker_thread in workers:
        worker_thread.join(timeout=5)
    
    # Collect results in frame order
    results_dict = {}
    
    while not result_queue.empty():
        try:
            result = result_queue.get_nowait()
            results_dict[result['frame_number']] = result
        except queue.Empty:
            break
    
    # Sort by frame number to maintain order
    detected_frames = [results_dict[fn] for fn in sorted(results_dict.keys())]
    
    print(f"\nDone! Queued {frames_queued} frames total.")
    print(f"Total detected frames with enemies: {len(detected_frames)}\n")
    
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
    print(f"\n{'='*60}")
    print(f"Multithreaded Detection (Workers: {num_workers}) - Total Time: {elapsed_time:.2f} seconds")
    print(f"{'='*60}\n")


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
    print(f"\n{'='*60}")
    print(f"Single-threaded Detection - Total Time: {elapsed_time:.2f} seconds")
    print(f"{'='*60}\n")


# Run the detection on a video file
import os
video_path = os.path.join(os.path.dirname(__file__), 'testVid.mp4')

# Run single-threaded version
print("Running single-threaded detection...\n")
detect_enemies_in_video_single_threaded(video_path, max_detected_frames_to_display=5)
print("Running multithreaded detection...\n")
# Run multithreaded version
detect_enemies_in_video(video_path, max_detected_frames_to_display=5, num_workers=4)