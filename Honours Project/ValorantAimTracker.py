from ultralyticsplus import YOLO, render_result
import torch
import ultralytics.nn.tasks
import ultralytics.nn.modules
import cv2
import os
import subprocess
import tempfile
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import queue
import threading
import time
import multiprocessing as mp
from multiprocessing import Pool

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Install with: pip install pytesseract")
    print("Also requires Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")

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

# Find enemy class indices
ENEMY_CLASS_INDICES = [idx for idx, name in model.names.items() if 'enemy' in name.lower()]


def _process_frame(args):
    """Worker function for multiprocessing pool that detects enemies in a frame."""
    frame_number, frame = args
    try:
        results = model.predict(frame, classes=ENEMY_CLASS_INDICES)
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
    frame_skip = 3  # Process every 3rd frame
    
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
    
    # Check crosshair placement relative to enemy head
    if detected_frames:
        detected_frames = check_crosshair_placement(detected_frames)
    
    # Measure reaction time from enemy appearance to shot
    analysis_results = None
    if detected_frames:
        analysis_results = measure_reaction_time(detected_frames, fps, frame_skip, video_path=video_path)
    
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
                print(f"    Crosshair Placement: {detection['crosshair_placement']}")
                print(f"    Vertical Offset: {detection['crosshair_y_diff']:.1f} pixels")
            
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
    return elapsed_time, analysis_results


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
            # Perform inference on this frame (only looking for enemies)
            results = model.predict(frame, classes=ENEMY_CLASS_INDICES)
            
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
    
    # Check crosshair placement relative to enemy head
    if detected_frames:
        detected_frames = check_crosshair_placement(detected_frames)
    
    # Measure reaction time from enemy appearance to shot
    analysis_results = None
    if detected_frames:
        analysis_results = measure_reaction_time(detected_frames, fps, frame_skip, video_path=video_path)
    
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
                print(f"    Crosshair Placement: {detection['crosshair_placement']}")
                print(f"    Vertical Offset: {detection['crosshair_y_diff']:.1f} pixels")
            
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
    return elapsed_time, analysis_results

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


def check_crosshair_placement(detected_frames, threshold=20):
    """
    Check if the crosshair is positioned higher than, lower than, or on level with enemy head.
    
    Args:
        detected_frames: List of detection dictionaries with crosshair_position and detections
        threshold: Pixel threshold for "on level" classification (default 20 pixels)
    
    Returns:
        List of detection dictionaries with added 'crosshair_placement' key for each detection
    """
    for detection_info in detected_frames:
        crosshair_x, crosshair_y = detection_info['crosshair_position']
        
        for detection in detection_info['detections']:
            box = detection['box']  # [x1, y1, x2, y2]
            
            # y1 is the top of the head (head level)
            head_level_y = box[1]
            
            # Determine placement relative to head
            diff = crosshair_y - head_level_y
            
            if diff < -threshold:
                placement = "HIGHER than head"
            elif diff > threshold:
                placement = "LOWER than head"
            else:
                placement = "ON LEVEL with head"
            
            detection['crosshair_placement'] = placement
            detection['head_level_y'] = head_level_y
            detection['crosshair_y_diff'] = diff
    
    return detected_frames


def measure_reaction_time(detected_frames, fps, frame_skip, ammo_region=(0.55, 0.88, 0.70, 0.99), video_path=None):
    """
    Measure reaction time from enemy appearance to first shot fired.
    Each engagement continues until ammo count changes (shot fired).
    
    Args:
        detected_frames: List of detection dictionaries with frame_number and detections
        fps: Frames per second of the video
        frame_skip: Number of frames skipped between processed frames
        ammo_region: Tuple (x1, y1, x2, y2) as fractions of image dimensions where ammo counter is located
    
    Returns:
        None - prints reaction time for each engagement
    """
    if not detected_frames:
        print("No enemy detections found.")
        return
    
    # Sort detected frames by frame number to ensure chronological order
    detected_frames_sorted = sorted(detected_frames, key=lambda x: x['frame_number'])
    
    # Extract ammo counts for all frames first
    ammo_counts = {}
    print("\nExtracting ammo count from frames for reaction time analysis...")
    successful_extractions = 0
    failed_extractions = 0
    
    for detection_info in detected_frames_sorted:
        ammo = extract_ammo_count(detection_info['frame_image'], ammo_region)
        ammo_counts[detection_info['frame_number']] = ammo
        if ammo is not None:
            successful_extractions += 1
        else:
            failed_extractions += 1
    
    print(f"Ammo extraction complete: {successful_extractions} successful, {failed_extractions} failed\n")
    
    # Extend ammo analysis beyond last detection to find final ammo change
    if video_path and detected_frames_sorted:
        last_detection_frame = detected_frames_sorted[-1]['frame_number']
        print(f"Last detection at frame {last_detection_frame}, scanning forward for ammo changes...\n")
        
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_detection_frame)
            frame_count = last_detection_frame
            last_ammo = ammo_counts.get(last_detection_frame)
            
            # Scan up to 10 seconds ahead for ammo changes
            scan_limit = last_detection_frame + int(10.0 * fps)
            
            while frame_count < scan_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check every frame skip
                if (frame_count - last_detection_frame) % frame_skip == 0:
                    ammo = extract_ammo_count(frame, ammo_region)
                    
                    if ammo is not None and ammo not in ammo_counts.values() or (last_ammo is not None and ammo != last_ammo):
                        ammo_counts[frame_count] = ammo
                        last_ammo = ammo
                
                frame_count += 1
            
            cap.release()
    
    # Group frames into engagements based on ammo changes (with timeouts)
    # Primary: ammo changes define engagement boundaries
    # Fallback: 3-second timeout or frame gaps define boundaries when ammo can't be read
    engagements = []
    current_engagement = []
    current_engagement_ammo = None
    current_engagement_start_frame = None
    frame_gap_threshold = frame_skip * 5
    timeout_frames = int(3.0 * fps / frame_skip)  # 3 seconds worth of detection frames
    
    for detection_info in detected_frames_sorted:
        frame_num = detection_info['frame_number']
        frame_ammo = ammo_counts.get(frame_num)
        
        if not current_engagement:
            # Start new engagement
            current_engagement = [detection_info]
            current_engagement_ammo = frame_ammo
            current_engagement_start_frame = frame_num
        else:
            last_frame_num = current_engagement[-1]['frame_number']
            frame_gap = frame_num - last_frame_num
            time_in_engagement = frame_num - current_engagement_start_frame
            
            # Check for 3-second timeout (absolute engagement duration limit)
            if time_in_engagement > int(3.0 * fps):
                # 3 seconds elapsed - end engagement
                engagements.append((current_engagement, current_engagement_ammo))
                current_engagement = [detection_info]
                current_engagement_ammo = frame_ammo
                current_engagement_start_frame = frame_num
            # Check for ammo change (primary method)
            elif frame_ammo is not None and current_engagement_ammo is not None:
                if frame_ammo != current_engagement_ammo:
                    # Ammo changed - add this frame to current engagement, then end it
                    current_engagement.append(detection_info)
                    engagements.append((current_engagement, frame_ammo))
                    # Don't start new engagement; let next detection do that
                    current_engagement = []
                    current_engagement_ammo = None
                    current_engagement_start_frame = None
                else:
                    # Ammo same - continue engagement
                    current_engagement.append(detection_info)
            # Check for frame gap (fallback when ammo unreadable)
            elif frame_gap > frame_gap_threshold:
                # Large gap indicates new engagement
                engagements.append((current_engagement, current_engagement_ammo))
                current_engagement = [detection_info]
                current_engagement_ammo = frame_ammo
                current_engagement_start_frame = frame_num
            else:
                # Continue current engagement
                current_engagement.append(detection_info)
    
    # Don't forget the last engagement
    if current_engagement:
        engagements.append((current_engagement, current_engagement_ammo))
    
    print(f"\nReaction Time Analysis: {len(engagements)} engagement(s) detected\n")
    
    # Collect analysis results
    reaction_times = []
    crosshair_placements = []
    engagements_with_shots = 0
    
    # Process each engagement
    for eng_idx, (engagement_frames, final_ammo) in enumerate(engagements, 1):
        print(f"--- Engagement {eng_idx} ---")
        
        # Get the first and last detection frame
        first_detection_frame = engagement_frames[0]['frame_number']
        last_detection_frame = engagement_frames[-1]['frame_number']
        
        # Get initial ammo at engagement start
        initial_ammo = ammo_counts.get(first_detection_frame)
        
        # Skip engagement if no initial ammo count
        if initial_ammo is None:
            print(f"SKIPPED: Could not extract ammo at engagement start")
            print()
            continue
        
        print(f"Enemy appeared at: Frame {first_detection_frame}")
        print(f"Initial ammo: {initial_ammo}")
        
        # Collect crosshair placements from all detections in engagement
        for detection_info in engagement_frames:
            for detection in detection_info['detections']:
                if 'crosshair_placement' in detection:
                    crosshair_placements.append(detection['crosshair_placement'])
        
        # Check ammo changes in first 100 frames
        print("\nFirst 100 frames of engagement:")
        ammo_changed_in_100 = False
        for frame_info in engagement_frames:
            frame_num = frame_info['frame_number']
            if frame_num - first_detection_frame <= 100:
                ammo = ammo_counts.get(frame_num)
                frames_since_start = frame_num - first_detection_frame
                if ammo is not None:
                    print(f"  Frame {frame_num} (+{frames_since_start}): {ammo}" + (" ✓ CHANGED" if ammo != initial_ammo else ""))
                    if ammo != initial_ammo:
                        ammo_changed_in_100 = True
                else:
                    print(f"  Frame {frame_num} (+{frames_since_start}): Failed to extract")
        
        if not ammo_changed_in_100:
            print("  → No ammo change detected in first 100 frames")
        
        print("\nFull engagement ammo readings:")
        for frame_info in engagement_frames:
            frame_num = frame_info['frame_number']
            ammo = ammo_counts.get(frame_num)
            print(f"  Frame {frame_num}: {ammo if ammo else 'Failed to extract'}")
        
        # Shot is detected when ammo changed
        if final_ammo is not None and final_ammo != initial_ammo:
            reaction_frames = last_detection_frame - first_detection_frame
            reaction_time_seconds = reaction_frames * (frame_skip / fps)
            reaction_times.append(reaction_time_seconds)
            engagements_with_shots += 1
            print(f"Shot fired at: Frame {last_detection_frame}")
            print(f"Ammo after shot: {final_ammo}")
            print(f"⚡ Reaction Time: {reaction_time_seconds:.3f} seconds ({reaction_frames} frames, {reaction_time_seconds*1000:.0f}ms)")
        else:
            print(f"No shot fired in this engagement (ammo did not change)")
        
        print()  # Blank line between engagements
    
    # Calculate averages
    avg_reaction_time = sum(reaction_times) / len(reaction_times) if reaction_times else 0
    
    # Calculate most common crosshair placement
    avg_crosshair_placement = None
    if crosshair_placements:
        placement_counts = {}
        for placement in crosshair_placements:
            placement_counts[placement] = placement_counts.get(placement, 0) + 1
        avg_crosshair_placement = max(placement_counts, key=placement_counts.get)
    
    # Summary statistics
    if detected_frames_sorted:
        first_frame = detected_frames_sorted[0]['frame_number']
        last_frame = detected_frames_sorted[-1]['frame_number']
        total_detection_frames = last_frame - first_frame + 1
        total_actual_frames = len(detected_frames_sorted)
        print(f"\n=== Summary ===")
        print(f"Total video frames analyzed: {total_detection_frames} (frames {first_frame} to {last_frame})")
        print(f"Total detection frames: {total_actual_frames}")
        print(f"Total engagements: {len(engagements)}")
        print(f"Engagements with shots: {engagements_with_shots}")
        print(f"Average reaction time: {avg_reaction_time:.3f}s")
        print(f"Most common crosshair placement: {avg_crosshair_placement}")
    
    # Return results dictionary
    return {
        'reaction_times': reaction_times,
        'avg_reaction_time': avg_reaction_time,
        'crosshair_placements': crosshair_placements,
        'avg_crosshair_placement': avg_crosshair_placement,
        'total_engagements': len(engagements),
        'engagements_with_shots': engagements_with_shots
    }


def extract_ammo_count(frame, ammo_region=(0.55, 0.88, 0.70, 0.99)):
    """
    Extract ammo count from frame using OCR on the HUD region.
    Region is tuned to capture the magazine ammo counter in Valorant's HUD.
    
    Args:
        frame: Input video frame (numpy array)
        ammo_region: Tuple (x1, y1, x2, y2) as fractions of image dimensions.
                    Default (0.68, 0.90, 0.76, 0.99) targets the magazine ammo display.
    
    Returns:
        String of extracted digits, or None if extraction fails
    """
    # Find Tesseract binary
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    
    tesseract_path = None
    for path in possible_paths:
        if os.path.exists(path):
            tesseract_path = path
            break
    
    if tesseract_path is None:
        return None

    try:
        height, width = frame.shape[:2]
        x1 = int(ammo_region[0] * width)
        y1 = int(ammo_region[1] * height)
        x2 = int(ammo_region[2] * width)
        y2 = int(ammo_region[3] * height)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        # Simple threshold - find the right balance
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        binary = cv2.medianBlur(binary, 3)

        # Save binary image to temp file and use Tesseract via subprocess
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, binary)
        
        # Also try inverted binary
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_inv_path = tmp.name
            inverted = cv2.bitwise_not(binary)
            cv2.imwrite(tmp_inv_path, inverted)
        
        try:
            # Try different PSM modes
            for psm in [6, 7, 11]:
                for img_path in [tmp_path, tmp_inv_path]:
                    result = subprocess.run(
                        [tesseract_path, img_path, 'stdout', '--psm', str(psm)],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='ignore',
                        timeout=10
                    )
                    
                    text = result.stdout.strip()
                    digits = ''.join(ch for ch in text if ch.isdigit())
                    
                    if digits:
                        return digits
            
            return None
        finally:
            # Clean up temp files
            for path in [tmp_path, tmp_inv_path]:
                if os.path.exists(path):
                    os.remove(path)
    except Exception as e:
        return None


# Run the detection on a video file
import os
video_path = os.path.join(os.path.dirname(__file__), 'testVid.mp4')

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