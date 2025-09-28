import subprocess
import os

def download_model(file_id, output):
    subprocess.run(["gdown", file_id, "-O", output], check=True)

def convert_to_mp4(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File don't exist: {input_path}")
    subprocess.run(["ffmpeg", "-i", input_path, "-vcodec", "libx264", output_path], check=True)