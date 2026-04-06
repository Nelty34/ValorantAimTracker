import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Install with: pip install pytesseract")


def extract_ammo_count(frame, ammo_region=(0.80, 0.82, 1.0, 1.0)):
    """
    Extract ammo count from frame using OCR on the HUD region.
    Default region is tuned for Valorant's ammo counter (bottom-right corner).
    
    Args:
        frame: Input video frame (numpy array)
        ammo_region: Tuple (x1, y1, x2, y2) as fractions of image dimensions where ammo is located
                    Default (0.80, 0.82, 1.0, 1.0) targets Valorant's ammo counter
    
    Returns:
        Integer ammo count, or None if extraction fails
    """
    if not TESSERACT_AVAILABLE:
        return None
    
    try:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = ammo_region
        x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
        
        # Crop the ammo region
        ammo_crop = frame[y1:y2, x1:x2]
        
        # Convert to grayscale and apply threshold for better OCR
        gray = cv2.cvtColor(ammo_crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Extract text using tesseract
        text = pytesseract.image_to_string(binary, config='--psm 6')
        
        # Extract first number found in text
        numbers = ''.join(filter(str.isdigit, text))
        if numbers:
            return int(numbers)
    except Exception as e:
        print(f"Error extracting ammo: {e}")
        pass
    
    return None


if __name__ == '__main__':
    # Load the image
    image_path = 'image.jpg'  # Make sure the captured image is named 'image.png'
    
    try:
        # Read image with cv2
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Error: Could not load image from {image_path}")
        else:
            print(f"Image loaded successfully. Shape: {frame.shape}")
            
            # Test with default region
            print("\nTesting default ammo region (0.80, 0.82, 1.0, 1.0):")
            ammo = extract_ammo_count(frame)
            print(f"Extracted ammo count: {ammo}")
            
            # Test with alternative regions to find optimal one
            test_regions = [
                ((0.75, 0.80, 1.0, 1.0), "Larger region (0.75, 0.80, 1.0, 1.0)"),
                ((0.80, 0.75, 1.0, 1.0), "Higher region (0.80, 0.75, 1.0, 1.0)"),
                ((0.70, 0.78, 1.0, 1.0), "Much larger region (0.70, 0.78, 1.0, 1.0)"),
            ]
            
            print("\nTesting alternative regions:")
            for region, description in test_regions:
                ammo = extract_ammo_count(frame, ammo_region=region)
                print(f"{description}: {ammo}")
            
            # Save a visualization of the crop region
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = int(0.80 * width), int(0.82 * height), int(1.0 * width), int(1.0 * height)
            ammo_crop = frame[y1:y2, x1:x2]
            cv2.imwrite('ammo_crop_preview.png', ammo_crop)
            print("\nSaved ammo crop preview to: ammo_crop_preview.png")
    
    except Exception as e:
        print(f"Error: {e}")
