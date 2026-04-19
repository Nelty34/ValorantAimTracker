# spectate_test.py
# Put your image filename below, then run:
# python spectate_test.py

import cv2
import os
import tempfile
import subprocess
from difflib import SequenceMatcher

# ==================================================
# PUT YOUR IMAGE FILE HERE
# ==================================================
IMAGE_FILE = "alive_test.png"

# Enable debug images
DEBUG = True


def find_tesseract_path():
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return "tesseract"


def ocr_text_from_image(image, psm=7):
    tesseract_path = find_tesseract_path()

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
                "-c",
                "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ "
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        return result.stdout.strip().upper()

    except Exception as e:
        print("OCR Error:", e)
        return ""

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def is_player_dead_spectating(
    frame,
    spectate_region=(0.00, 0.70, 0.34, 0.86),
    similarity_threshold=0.72,
    debug=False
):
    h, w = frame.shape[:2]

    x1 = int(spectate_region[0] * w)
    y1 = int(spectate_region[1] * h)
    x2 = int(spectate_region[2] * w)
    y2 = int(spectate_region[3] * h)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return False

    rh, rw = roi.shape[:2]

    # lower-right area where SWITCH PLAYER text should be
    text_roi = roi[int(rh * 0.35):rh, int(rw * 0.28):rw]

    gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    binary_inv = cv2.bitwise_not(binary)

    candidates = []

    for img in [binary, binary_inv]:
        for psm in [7, 6, 11]:
            text = ocr_text_from_image(img, psm=psm)
            if text:
                candidates.append(text)

    target = "SWITCHPLAYER"

    best_score = 0
    best_text = ""

    for text in candidates:
        cleaned = "".join(c for c in text if c.isalpha())

        if not cleaned:
            continue

        score = SequenceMatcher(None, cleaned, target).ratio()

        if target in cleaned:
            score = 1.0

        if score > best_score:
            best_score = score
            best_text = cleaned

    if debug:
        print("OCR Candidates:", candidates)
        print("Best Match:", best_text)
        print("Score:", round(best_score, 3))

        cv2.imwrite("debug_roi.png", roi)
        cv2.imwrite("debug_text_roi.png", text_roi)
        cv2.imwrite("debug_binary.png", binary)

    return best_score >= similarity_threshold


# ==================================================
# MAIN
# ==================================================

frame = cv2.imread(IMAGE_FILE)

if frame is None:
    print("Could not load:", IMAGE_FILE)
    quit()

dead = is_player_dead_spectating(frame, debug=DEBUG)

print()
print("RESULT")
print("------------------")

if dead:
    print("PLAYER IS DEAD / SPECTATING")
else:
    print("PLAYER IS ALIVE")

print()