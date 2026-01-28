import subprocess
from datetime import datetime
import os

def screen_record(output_file=None, duration=None):
    """
    Record the full screen using FFmpeg.
    
    Args:
        output_file: Output video file path (default: recording_TIMESTAMP.mp4)
        duration: Recording duration in seconds (None for unlimited)
    """
    if output_file is None:
        recordings_dir = os.path.join(os.path.dirname(__file__), "recordings")
        os.makedirs(recordings_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(recordings_dir, f"recording_{timestamp}.mp4")
    
    try:
        cmd = [
            "ffmpeg",
            "-f", "gdigrab",
            "-i", "desktop",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-profile:v", "baseline",
            "-level", "3.1",
            "-pix_fmt", "yuv420p",
        ]
        
        if duration:
            cmd.extend(["-t", str(duration)])
        
        cmd.extend(["-y", output_file])
        
        subprocess.run(cmd, stdin=subprocess.DEVNULL, check=True, timeout=duration + 10 if duration else None)
        print(f"Recording saved to {output_file}")
    
    except FileNotFoundError:
        print("FFmpeg not found. Install it using: choco install ffmpeg (or download from https://ffmpeg.org/download.html")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    screen_record(duration=10)