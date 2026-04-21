from difflib import SequenceMatcher
import os
import subprocess
import tempfile
import time
import multiprocessing as mp

import cv2
import numpy as np
from ultralyticsplus import YOLO
import torch

# Monkeypatch torch.load so that, when weights_only is not provided
# it defaults to False
if not hasattr(torch.load, '_is_patched'):
    _original_torch_load = torch.load

    def _load_with_weights_only_default_false(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    _load_with_weights_only_default_false._is_patched = True
    torch.load = _load_with_weights_only_default_false

# Global constants
MODEL_NAME = 'keremberke/yolov8m-valorant-detection'
DETECTION_CONFIDENCE = 0.8
DETECTION_IOU = 0.45
MAX_DETECTIONS = 1000
DEFAULT_FRAME_SKIP_MULTI = 3
DEFAULT_FRAME_SKIP_SINGLE = 5
DEFAULT_AMMO_REGION = (0.55, 0.88, 0.70, 0.99)
DEFAULT_SPECTATE_REGION = (0.00, 0.70, 0.34, 0.86)
DEAD_CHECK_INTERVAL = 10
DEFAULT_VIDEO_NAME = 'dead_test.mp4'

# Initialize the YOLO model and set overrides for detection parameters
model = YOLO(MODEL_NAME)
model.overrides['conf'] = DETECTION_CONFIDENCE
model.overrides['iou'] = DETECTION_IOU
model.overrides['agnostic_nms'] = False
model.overrides['max_det'] = MAX_DETECTIONS

# Only 
ENEMY_CLASS_INDICES = [idx for idx, name in model.names.items() if 'enemy' in name.lower()]

# Ensure Tesseract OCR is available and can be found
def _find_tesseract_path():
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# OCR function that processes an image and returns uppercase text, with error handling
def _ocr_text_from_image(image, psm=7, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
    tesseract_path = _find_tesseract_path()
    if tesseract_path is None:
        return ""
    # Use a temporary file to avoid issues with passing image data directly to Tesseract
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image)

        result = subprocess.run(
            [
                tesseract_path,
                tmp_path,
                "stdout",
                "--psm", str(psm),
                "-c", f"tessedit_char_whitelist={whitelist}"
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
        )
        return result.stdout.strip().upper()
    except Exception:
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# Function to detect if the player is dead/spectating based on the spectate banner in the bottom-left corner
def is_player_dead(frame, spectate_region=DEFAULT_SPECTATE_REGION, template_path=None,
                   ocr_similarity_threshold=0.72, template_threshold=0.70):
    # Extract the spectate banner region from the frame
    h, w = frame.shape[:2]
    x1 = int(spectate_region[0] * w)
    y1 = int(spectate_region[1] * h)
    x2 = int(spectate_region[2] * w)
    y2 = int(spectate_region[3] * h)

    # Check if the region is valid and not empty
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    # Focus on the area where the "SWITCH PLAYER" text appears
    rh, rw = roi.shape[:2]
    text_roi = roi[int(rh * 0.40):rh, int(rw * 0.32):rw]
    # Preprocess the image for OCR: convert to grayscale, resize, and apply thresholding
    gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    binary_inv = cv2.bitwise_not(binary)
    # Define the target text and initialize the best score
    target = "SWITCHPLAYER"
    best_score = 0.0
    # Perform OCR on both the binary and inverted images using different PSM modes, and calculate similarity scores
    for img in (binary, binary_inv):
        for psm in (7, 6, 11):
            text = _ocr_text_from_image(img, psm=psm)
            normalized = "".join(ch for ch in text if ch.isalpha())
            if not normalized:
                continue

            score = SequenceMatcher(None, normalized, target).ratio()
            if target in normalized:
                score = 1.0
            best_score = max(best_score, score)
    # Determine if the best OCR similarity score meets the threshold to classify the player as dead/spectating
    if best_score >= ocr_similarity_threshold:
        return True

    return False

# Worker function for multiprocessing pool that detects enemies in a frame and returns structured detection info
def _process_frame(args):
    frame_number, frame = args
    # Run the YOLO model on the frame to detect enemies, and structure the results with frame metadata and detection details
    try:
        results = model.predict(frame, classes=ENEMY_CLASS_INDICES)
        enemy_detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                if 'enemy' in class_name.lower() and confidence >= DETECTION_CONFIDENCE:
                    enemy_detections.append({
                        'frame': frame_number,
                        'class': class_name,
                        'confidence': confidence,
                        'box': box.xyxy[0].cpu().numpy(),
                    })
        # Only return detection info if enemies were detected in the frame
        if not enemy_detections:
            return None
        # Get frame dimensions for crosshair position calculations
        height, width = frame.shape[:2]
        return {
            'frame_number': frame_number,
            'frame_height': height,
            'frame_width': width,
            'detections': enemy_detections,
        }
    except Exception:
        return None

# Function to calculate the crosshair position based on frame dimensions, and add it to detection info for each frame
def calculate_crosshair_position(frame_or_frames):
    # If input is a single frame (numpy array), return the center coordinates as the crosshair position
    if isinstance(frame_or_frames, np.ndarray):
        height, width = frame_or_frames.shape[:2]
        return width // 2, height // 2
    # If input is a list of detection info dictionaries, calculate the crosshair position for each frame and add it to the dictionary
    if isinstance(frame_or_frames, list):
        for detection_info in frame_or_frames:
            if 'frame_height' in detection_info and 'frame_width' in detection_info:
                height = detection_info['frame_height']
                width = detection_info['frame_width']
            elif 'frame_image' in detection_info:
                frame = detection_info['frame_image']
                height, width = frame.shape[:2]
            else:
                raise KeyError("Detection info missing frame dimensions")

            detection_info['crosshair_position'] = (width // 2, height // 2)
        return frame_or_frames

    raise TypeError("Input must be a numpy array or list of detection dictionaries")

# Function to check the placement of the crosshair relative to detected enemy head positions, and classify it as higher, lower, or on level with the head
def check_crosshair_placement(detected_frames, threshold=20):
    # For each detected frame, compare the y-coordinate of the crosshair with the y-coordinate of the detected enemy head
    for detection_info in detected_frames:
        _, crosshair_y = detection_info['crosshair_position']

        for detection in detection_info['detections']:
            box = detection['box']
            head_level_y = box[1]
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

# Function to extract the ammo count from a specified HUD region using OCR
def extract_ammo_count(frame, ammo_region=DEFAULT_AMMO_REGION):
    tesseract_path = _find_tesseract_path()
    if tesseract_path is None:
        return None
    # Calculate the pixel coordinates of the ammo region based on the frame dimensions and the specified relative region
    try:
        height, width = frame.shape[:2]
        x1 = int(ammo_region[0] * width)
        y1 = int(ammo_region[1] * height)
        x2 = int(ammo_region[2] * width)
        y2 = int(ammo_region[3] * height)
        # Check if the region is valid and not empty
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        # Preprocess the cropped image for OCR: convert to grayscale, resize, and apply thresholding to enhance digit visibility
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        binary = cv2.medianBlur(binary, 3)
        inverted = cv2.bitwise_not(binary)

        tmp_files = []
        for img in (binary, inverted):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cv2.imwrite(tmp.name, img)
                tmp_files.append(tmp.name)
        # Try multiple PSM modes to improve OCR accuracy, and extract only digits from the OCR result to determine the ammo count
        try:
            for psm in (6, 7, 11):
                for img_path in tmp_files:
                    result = subprocess.run(
                        [tesseract_path, img_path, 'stdout', '--psm', str(psm)],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='ignore',
                        timeout=10,
                    )
                    digits = ''.join(ch for ch in result.stdout.strip() if ch.isdigit())
                    if digits:
                        return digits
            return None
        finally:
            for path in tmp_files:
                if os.path.exists(path):
                    os.remove(path)
    except Exception:
        return None

# Function to measure reaction time from the first enemy appearance to the first shot fired, using detected frames and ammo count changes
def measure_reaction_time(detected_frames, fps, frame_skip, ammo_region=DEFAULT_AMMO_REGION, video_path=None):
    if not detected_frames:
        return None
    # Sort detected frames by frame number
    detected_frames_sorted = sorted(detected_frames, key=lambda x: x['frame_number'])
    ammo_counts = {}
    # If a video path is provided, open the video and extract the ammo count for each detected frame using OCR, storing the results in a dictionary
    cap_ammo = cv2.VideoCapture(video_path) if video_path else None
    if cap_ammo is not None and not cap_ammo.isOpened():
        cap_ammo = None
    # Loop through the detected frames, seek to the corresponding frame in the video, and extract the ammo count using OCR for each frame, storing it in the ammo_counts dictionary
    for detection_info in detected_frames_sorted:
        frame = None
        if cap_ammo is not None:
            frame_num = detection_info['frame_number']
            cap_ammo.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap_ammo.read()
            if not ret:
                frame = None
        # Extract the ammo count for the current frame using OCR and store it in the ammo_counts dictionary, keyed by frame number
        ammo_counts[detection_info['frame_number']] = extract_ammo_count(frame, ammo_region) if frame is not None else None
    # Release the video capture object used for ammo extraction if it was opened
    if cap_ammo is not None:
        cap_ammo.release()
    # If a video path is provided, continue scanning forward from the last detected frame for a short duration to capture any ammo changes
        last_detection_frame = detected_frames_sorted[-1]['frame_number']
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_detection_frame)
            frame_count = last_detection_frame
            last_ammo = ammo_counts.get(last_detection_frame)
            scan_limit = last_detection_frame + int(10.0 * fps)
            # Loop through the frames after the last detected frame, extracting the ammo count at regular intervals, and store any changes in the ammo count in the ammo_counts dictionary
            while frame_count < scan_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                # Only check for ammo changes at intervals defined by frame_skip to reduce processing load, and store any new ammo counts that differ from the last known count
                if (frame_count - last_detection_frame) % frame_skip == 0:
                    ammo = extract_ammo_count(frame, ammo_region)
                    if ammo is not None and (ammo not in ammo_counts.values() or (last_ammo is not None and ammo != last_ammo)):
                        ammo_counts[frame_count] = ammo
                        last_ammo = ammo

                frame_count += 1
            cap.release()
    # Group detected frames into engagements based on time gaps and ammo count changes, and calculate reaction times and crosshair placements for each engagement
    engagements = []
    current_engagement = []
    current_engagement_ammo = None
    current_engagement_start_frame = None
    frame_gap_threshold = frame_skip * 5
    # Loop through the sorted detected frames and group them into engagements based on time gaps between detections and changes in ammo count, storing the grouped engagements along with their associated ammo counts
    for detection_info in detected_frames_sorted:
        frame_num = detection_info['frame_number']
        frame_ammo = ammo_counts.get(frame_num)
        # If there is no current engagement, start a new one with the current detection info and ammo count
        if not current_engagement:
            current_engagement = [detection_info]
            current_engagement_ammo = frame_ammo
            current_engagement_start_frame = frame_num
            continue
        # Calculate the time gap between the current detection and the last detection in the current engagement, as well as the time elapsed since the start of the engagement
        last_frame_num = current_engagement[-1]['frame_number']
        frame_gap = frame_num - last_frame_num
        time_in_engagement = frame_num - current_engagement_start_frame
        # If the time in the current engagement exceeds a certain threshold, or if there is a change in ammo count, or if there is a large gap between detections, close the current engagement and start a new one with the current detection info and ammo count
        if time_in_engagement > int(3.0 * fps):
            engagements.append((current_engagement, current_engagement_ammo))
            current_engagement = [detection_info]
            current_engagement_ammo = frame_ammo
            current_engagement_start_frame = frame_num
        # If there is a change in ammo count, close the current engagement and start a new one with the current detection info and ammo count
        elif frame_ammo is not None and current_engagement_ammo is not None:
            if frame_ammo != current_engagement_ammo:
                current_engagement.append(detection_info)
                engagements.append((current_engagement, frame_ammo))
                current_engagement = []
                current_engagement_ammo = None
                current_engagement_start_frame = None
            else:
                current_engagement.append(detection_info)
        # If there is a large gap between detections, close the current engagement and start a new one with the current detection info and ammo count
        elif frame_gap > frame_gap_threshold:
            engagements.append((current_engagement, current_engagement_ammo))
            current_engagement = [detection_info]
            current_engagement_ammo = frame_ammo
            current_engagement_start_frame = frame_num
        else:
            current_engagement.append(detection_info)
    # After looping through all detected frames, if there is an open engagement that has not been closed, close it and add it to the engagements list
    if current_engagement:
        engagements.append((current_engagement, current_engagement_ammo))

    reaction_times = []
    crosshair_placements = []
    engagements_with_shots = 0
    # Loop through the grouped engagements and calculate reaction times based on the time from the first detection to the first ammo change, as well as collect crosshair placement data for each engagement, and count how many engagements resulted in shots being fired
    for engagement_frames, final_ammo in engagements:
        first_detection_frame = engagement_frames[0]['frame_number']
        last_detection_frame = engagement_frames[-1]['frame_number']
        initial_ammo = ammo_counts.get(first_detection_frame)

        if initial_ammo is None:
            continue

        for detection_info in engagement_frames:
            for detection in detection_info['detections']:
                if 'crosshair_placement' in detection:
                    crosshair_placements.append(detection['crosshair_placement'])

        if final_ammo is not None and final_ammo != initial_ammo:
            reaction_frames = last_detection_frame - first_detection_frame
            reaction_times.append(reaction_frames * (frame_skip / fps))
            engagements_with_shots += 1
    # Filter out reaction times that are above a certain threshold to remove outliers, and calculate the average reaction time from the filtered list
    filtered_reaction_times = [rt for rt in reaction_times if rt <= 0.9]
    avg_reaction_time = sum(filtered_reaction_times) / len(filtered_reaction_times) if filtered_reaction_times else 0
    # Calculate the average crosshair placement by counting the occurrences of each placement category and selecting the one with the highest count, if any placements were recorded
    avg_crosshair_placement = None
    if crosshair_placements:
        placement_counts = {}
        for placement in crosshair_placements:
            placement_counts[placement] = placement_counts.get(placement, 0) + 1
        avg_crosshair_placement = max(placement_counts, key=placement_counts.get)
    # Return a dictionary containing the calculated reaction times, average reaction time, crosshair placements, average crosshair placement, total number of engagements, and number of engagements that resulted in shots being fired
    return {
        'reaction_times': reaction_times,
        'avg_reaction_time': avg_reaction_time,
        'crosshair_placements': crosshair_placements,
        'avg_crosshair_placement': avg_crosshair_placement,
        'total_engagements': len(engagements),
        'engagements_with_shots': engagements_with_shots,
    }

# Main function to run enemy detection on a video with optional multiprocessing, and return the elapsed time and analysis results
def _run_detection(video_path, frame_skip, max_detected_frames_to_display=10, num_workers=None,
                   chunk_size=50, frame_downsample=1):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file at {video_path}")
    # Get the frames per second (FPS) of the video for later use in reaction time calculations
    fps = cap.get(cv2.CAP_PROP_FPS)
    detected_frames = []
    processed_frame_index = 0
    last_dead_state = False
    # If multiprocessing is enabled and more than one worker is specified, use a multiprocessing pool to process frames in parallel, otherwise process frames sequentially in the main thread
    if num_workers and num_workers > 1:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_workers) as pool:
            frame_count = 0
            while True:
                frames_to_process = []
                while len(frames_to_process) < chunk_size:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Only process frames at intervals defined by frame_skip to reduce processing load, and check the player's dead/spectating state at regular intervals defined by DEAD_CHECK_INTERVAL, skipping enemy detection for frames where the player is classified as dead/spectating
                    if frame_count % frame_skip == 0:
                        if processed_frame_index % DEAD_CHECK_INTERVAL == 0:
                            last_dead_state = is_player_dead(frame)
                        # If the player is not dead/spectating, add the frame to the list of frames to be processed by the multiprocessing pool
                            if frame_downsample > 1:
                                h, w = frame.shape[:2]
                                frame = cv2.resize(frame, (w // frame_downsample, h // frame_downsample))
                            frames_to_process.append((frame_count, frame))

                        processed_frame_index += 1

                    frame_count += 1
                # If there are no frames to process continue to the next iteration of the loop, and if the end of the video is reached, break out of the loop
                if not frames_to_process:
                    if not ret:
                        break
                    continue
                # Use the multiprocessing pool to process the collected frames in parallel, and collect the results as they are completed, adding any non-None results to the detected_frames list
                results = pool.imap_unordered(
                    _process_frame,
                    frames_to_process,
                    chunksize=max(1, len(frames_to_process) // max(1, num_workers)),
                )
                for result in results:
                    if result is not None:
                        detected_frames.append(result)
    # If multiprocessing is not enabled, process frames sequentially in the main thread, checking the player's dead/spectating state at regular intervals and only performing enemy detection on frames where the player is classified as alive
    else:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                if processed_frame_index % DEAD_CHECK_INTERVAL == 0:
                    last_dead_state = is_player_dead(frame)

                if not last_dead_state:
                    results = model.predict(frame, classes=ENEMY_CLASS_INDICES)
                    enemy_detections = []
                    for result in results:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            confidence = float(box.conf[0])
                            if 'enemy' in class_name.lower() and confidence >= DETECTION_CONFIDENCE:
                                enemy_detections.append({
                                    'frame': frame_count,
                                    'class': class_name,
                                    'confidence': confidence,
                                    'box': box.xyxy[0].cpu().numpy(),
                                })
                    if enemy_detections:
                        height, width = frame.shape[:2]
                        detected_frames.append({
                            'frame_number': frame_count,
                            'frame_height': height,
                            'frame_width': width,
                            'detections': enemy_detections,
                        })

                processed_frame_index += 1

            frame_count += 1

    cap.release()
    detected_frames.sort(key=lambda x: x['frame_number'])
    # If any detected frames were found, calculate the crosshair position for each frame and check the placement of the crosshair relative to detected enemy head positions, adding this information to the detection results
    if detected_frames:
        calculate_crosshair_position(detected_frames)
        check_crosshair_placement(detected_frames)
    # If any detected frames were found, measure the reaction time from the first enemy appearance to the first shot fired using the detected frames and ammo count changes, and return the elapsed time and analysis results
    analysis_results = measure_reaction_time(detected_frames, fps, frame_skip, video_path=video_path) if detected_frames else None
    return time.time() - start_time, analysis_results

# Public function to detect enemies in a video using multiprocessing, with parameters for maximum detected frames to display, number of worker processes, chunk size for processing, and frame downsampling factor
def detect_enemies_in_video(video_path, max_detected_frames_to_display=10, num_workers=4,
                            chunk_size=50, frame_downsample=1):
    return _run_detection(
        video_path,
        frame_skip=DEFAULT_FRAME_SKIP_MULTI,
        max_detected_frames_to_display=max_detected_frames_to_display,
        num_workers=num_workers,
        chunk_size=chunk_size,
        frame_downsample=frame_downsample,
    )
# If this script is run directly, execute the enemy detection on the specified video path with multiprocessing enabled, and print the total elapsed time and analysis results
if __name__ == '__main__':
    video_path = os.path.join(os.path.dirname(__file__), DEFAULT_VIDEO_NAME)
    num_cores = mp.cpu_count()
    chunk_size = 50
    frame_downsample = 1

    elapsed, analysis_results = detect_enemies_in_video(
        video_path,
        max_detected_frames_to_display=25,
        num_workers=4,
        chunk_size=chunk_size,
        frame_downsample=frame_downsample,
    )

    print(f"Total Time: {elapsed:.2f} seconds")
    if analysis_results is not None:
        print(analysis_results)
