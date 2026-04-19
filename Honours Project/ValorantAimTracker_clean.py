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
import ultralytics.nn.tasks  # noqa: F401
import ultralytics.nn.modules  # noqa: F401


# Monkeypatch torch.load so that, when weights_only is not provided,
# it defaults to False (changes behavior at runtime only).
if not hasattr(torch.load, '_is_patched'):
    _original_torch_load = torch.load

    def _load_with_weights_only_default_false(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    _load_with_weights_only_default_false._is_patched = True
    torch.load = _load_with_weights_only_default_false


MODEL_NAME = 'keremberke/yolov8m-valorant-detection'
DETECTION_CONFIDENCE = 0.8
DETECTION_IOU = 0.45
MAX_DETECTIONS = 1000
DEFAULT_FRAME_SKIP_MULTI = 3
DEFAULT_FRAME_SKIP_SINGLE = 5
DEFAULT_AMMO_REGION = (0.55, 0.88, 0.70, 0.99)
DEFAULT_SPECTATE_REGION = (0.00, 0.70, 0.34, 0.86)
DEAD_CHECK_INTERVAL = 10
DEFAULT_VIDEO_NAME = 'twig1.mp4'


model = YOLO(MODEL_NAME)
model.overrides['conf'] = DETECTION_CONFIDENCE
model.overrides['iou'] = DETECTION_IOU
model.overrides['agnostic_nms'] = False
model.overrides['max_det'] = MAX_DETECTIONS

ENEMY_CLASS_INDICES = [idx for idx, name in model.names.items() if 'enemy' in name.lower()]


def _find_tesseract_path():
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def _ocr_text_from_image(image, psm=7, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
    """Run OCR on an image and return uppercase text or an empty string."""
    tesseract_path = _find_tesseract_path()
    if tesseract_path is None:
        return ""

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


def is_player_dead(frame, spectate_region=DEFAULT_SPECTATE_REGION, template_path=None,
                   ocr_similarity_threshold=0.72, template_threshold=0.70):
    """Detect whether the player is dead/spectating using the bottom-left spectate banner."""
    h, w = frame.shape[:2]
    x1 = int(spectate_region[0] * w)
    y1 = int(spectate_region[1] * h)
    x2 = int(spectate_region[2] * w)
    y2 = int(spectate_region[3] * h)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    rh, rw = roi.shape[:2]
    text_roi = roi[int(rh * 0.40):rh, int(rw * 0.32):rw]

    gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    binary_inv = cv2.bitwise_not(binary)

    target = "SWITCHPLAYER"
    best_score = 0.0

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

    if best_score >= ocr_similarity_threshold:
        return True

    if template_path is not None and os.path.exists(template_path):
        template = cv2.imread(template_path)
        if template is not None:
            tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
            best_template_score = 0.0

            for scale in (0.8, 0.9, 1.0, 1.1, 1.2):
                scaled_tpl = cv2.resize(tpl_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                th, tw = scaled_tpl.shape[:2]
                rh2, rw2 = roi_gray.shape[:2]
                if th > rh2 or tw > rw2:
                    continue

                result = cv2.matchTemplate(roi_gray, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                best_template_score = max(best_template_score, max_val)

            if best_template_score >= template_threshold:
                return True

    return False


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
                if 'enemy' in class_name.lower() and confidence >= DETECTION_CONFIDENCE:
                    enemy_detections.append({
                        'frame': frame_number,
                        'class': class_name,
                        'confidence': confidence,
                        'box': box.xyxy[0].cpu().numpy(),
                    })

        if not enemy_detections:
            return None

        height, width = frame.shape[:2]
        return {
            'frame_number': frame_number,
            'frame_height': height,
            'frame_width': width,
            'detections': enemy_detections,
        }
    except Exception:
        return None


def calculate_crosshair_position(frame_or_frames):
    """Calculate the centre pixel position (crosshair) for a frame or list of detections."""
    if isinstance(frame_or_frames, np.ndarray):
        height, width = frame_or_frames.shape[:2]
        return width // 2, height // 2

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


def check_crosshair_placement(detected_frames, threshold=20):
    """Classify crosshair placement relative to enemy head."""
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


def extract_ammo_count(frame, ammo_region=DEFAULT_AMMO_REGION):
    """Extract magazine ammo count from a HUD region using OCR."""
    tesseract_path = _find_tesseract_path()
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
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        binary = cv2.medianBlur(binary, 3)
        inverted = cv2.bitwise_not(binary)

        tmp_files = []
        for img in (binary, inverted):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cv2.imwrite(tmp.name, img)
                tmp_files.append(tmp.name)

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


def measure_reaction_time(detected_frames, fps, frame_skip, ammo_region=DEFAULT_AMMO_REGION, video_path=None):
    """Measure reaction time from first enemy appearance to first shot fired."""
    if not detected_frames:
        return None

    detected_frames_sorted = sorted(detected_frames, key=lambda x: x['frame_number'])
    ammo_counts = {}

    cap_ammo = cv2.VideoCapture(video_path) if video_path else None
    if cap_ammo is not None and not cap_ammo.isOpened():
        cap_ammo = None

    for detection_info in detected_frames_sorted:
        frame = None
        if cap_ammo is not None:
            frame_num = detection_info['frame_number']
            cap_ammo.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap_ammo.read()
            if not ret:
                frame = None

        ammo_counts[detection_info['frame_number']] = extract_ammo_count(frame, ammo_region) if frame is not None else None

    if cap_ammo is not None:
        cap_ammo.release()

    if video_path and detected_frames_sorted:
        last_detection_frame = detected_frames_sorted[-1]['frame_number']
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_detection_frame)
            frame_count = last_detection_frame
            last_ammo = ammo_counts.get(last_detection_frame)
            scan_limit = last_detection_frame + int(10.0 * fps)

            while frame_count < scan_limit:
                ret, frame = cap.read()
                if not ret:
                    break

                if (frame_count - last_detection_frame) % frame_skip == 0:
                    ammo = extract_ammo_count(frame, ammo_region)
                    if ammo is not None and (ammo not in ammo_counts.values() or (last_ammo is not None and ammo != last_ammo)):
                        ammo_counts[frame_count] = ammo
                        last_ammo = ammo

                frame_count += 1
            cap.release()

    engagements = []
    current_engagement = []
    current_engagement_ammo = None
    current_engagement_start_frame = None
    frame_gap_threshold = frame_skip * 5

    for detection_info in detected_frames_sorted:
        frame_num = detection_info['frame_number']
        frame_ammo = ammo_counts.get(frame_num)

        if not current_engagement:
            current_engagement = [detection_info]
            current_engagement_ammo = frame_ammo
            current_engagement_start_frame = frame_num
            continue

        last_frame_num = current_engagement[-1]['frame_number']
        frame_gap = frame_num - last_frame_num
        time_in_engagement = frame_num - current_engagement_start_frame

        if time_in_engagement > int(3.0 * fps):
            engagements.append((current_engagement, current_engagement_ammo))
            current_engagement = [detection_info]
            current_engagement_ammo = frame_ammo
            current_engagement_start_frame = frame_num
        elif frame_ammo is not None and current_engagement_ammo is not None:
            if frame_ammo != current_engagement_ammo:
                current_engagement.append(detection_info)
                engagements.append((current_engagement, frame_ammo))
                current_engagement = []
                current_engagement_ammo = None
                current_engagement_start_frame = None
            else:
                current_engagement.append(detection_info)
        elif frame_gap > frame_gap_threshold:
            engagements.append((current_engagement, current_engagement_ammo))
            current_engagement = [detection_info]
            current_engagement_ammo = frame_ammo
            current_engagement_start_frame = frame_num
        else:
            current_engagement.append(detection_info)

    if current_engagement:
        engagements.append((current_engagement, current_engagement_ammo))

    reaction_times = []
    crosshair_placements = []
    engagements_with_shots = 0

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

    filtered_reaction_times = [rt for rt in reaction_times if rt <= 0.9]
    avg_reaction_time = sum(filtered_reaction_times) / len(filtered_reaction_times) if filtered_reaction_times else 0

    avg_crosshair_placement = None
    if crosshair_placements:
        placement_counts = {}
        for placement in crosshair_placements:
            placement_counts[placement] = placement_counts.get(placement, 0) + 1
        avg_crosshair_placement = max(placement_counts, key=placement_counts.get)

    return {
        'reaction_times': reaction_times,
        'avg_reaction_time': avg_reaction_time,
        'crosshair_placements': crosshair_placements,
        'avg_crosshair_placement': avg_crosshair_placement,
        'total_engagements': len(engagements),
        'engagements_with_shots': engagements_with_shots,
    }


def _run_detection(video_path, frame_skip, max_detected_frames_to_display=10, num_workers=None,
                   chunk_size=50, frame_downsample=1):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    detected_frames = []
    processed_frame_index = 0
    last_dead_state = False

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

                    if frame_count % frame_skip == 0:
                        if processed_frame_index % DEAD_CHECK_INTERVAL == 0:
                            last_dead_state = is_player_dead(frame)

                        if not last_dead_state:
                            if frame_downsample > 1:
                                h, w = frame.shape[:2]
                                frame = cv2.resize(frame, (w // frame_downsample, h // frame_downsample))
                            frames_to_process.append((frame_count, frame))

                        processed_frame_index += 1

                    frame_count += 1

                if not frames_to_process:
                    if not ret:
                        break
                    continue

                results = pool.imap_unordered(
                    _process_frame,
                    frames_to_process,
                    chunksize=max(1, len(frames_to_process) // max(1, num_workers)),
                )
                for result in results:
                    if result is not None:
                        detected_frames.append(result)
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

    if detected_frames:
        calculate_crosshair_position(detected_frames)
        check_crosshair_placement(detected_frames)

    analysis_results = measure_reaction_time(detected_frames, fps, frame_skip, video_path=video_path) if detected_frames else None
    return time.time() - start_time, analysis_results


def detect_enemies_in_video(video_path, max_detected_frames_to_display=10, num_workers=4,
                            chunk_size=50, frame_downsample=1):
    """Multiprocessing version of video enemy detection."""
    return _run_detection(
        video_path,
        frame_skip=DEFAULT_FRAME_SKIP_MULTI,
        max_detected_frames_to_display=max_detected_frames_to_display,
        num_workers=num_workers,
        chunk_size=chunk_size,
        frame_downsample=frame_downsample,
    )


def detect_enemies_in_video_single_threaded(video_path, max_detected_frames_to_display=10):
    """Single-threaded version of video enemy detection."""
    return _run_detection(
        video_path,
        frame_skip=DEFAULT_FRAME_SKIP_SINGLE,
        max_detected_frames_to_display=max_detected_frames_to_display,
        num_workers=None,
    )


if __name__ == '__main__':
    video_path = os.path.join(os.path.dirname(__file__), DEFAULT_VIDEO_NAME)
    num_cores = mp.cpu_count()
    chunk_size = 50
    frame_downsample = 1

    if num_cores < 4:
        elapsed, analysis_results = detect_enemies_in_video_single_threaded(
            video_path,
            max_detected_frames_to_display=5,
        )
    else:
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
