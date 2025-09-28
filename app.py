import subprocess
import streamlit as st
import tempfile
import os
from yolov8_deepsort import Detector, DeepSORT, video_tracking
from utils import download_model, convert_to_mp4

# Load 'best.pt' model if don't
if not os.path.exists('best.pt'):
    download_model("1chUu0ksS2_4K0J8VgMzdwYVKwqTPxAmm", "best.pt")

if not os.path.exists('resources/networks/mars-small128.pb'):
    # Folder trên Google Drive
    subprocess.run([
        "gdown", "--no-check-certificate", "--folder",
        "https://drive.google.com/drive/folders/18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp"
    ], check=True)

# Streamlit UI
st.title("Pedestrian Tracking and Counting")
st.write("Upload a video to track and count pedestrians using YOLOv8 and DeepSORT.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "webm"])

if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'result_path' not in st.session_state:
    st.session_state.result_path = None
if 'total_unique_pedestrians' not in st.session_state:
    st.session_state.total_unique_pedestrians = 0

if uploaded_file is not None and not st.session_state.processing:
    with st.spinner("Processing video..."):
        # Save temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # Initialize the model
        detector = Detector("best.pt")
        tracker = DeepSORT()

        # Run tracking
        st.session_state.processing = True
        _, total_unique_pedestrians, save_result_path = video_tracking(
            video_path=video_path,
            detector=detector,
            tracker=tracker,
            is_save_result=True
        )

        os.makedirs("tracking_results", exist_ok=True)
        
        # Convert sang MP4
        compressed_video_path = 'tracking_results/result_compressed.mp4'
        convert_to_mp4(save_result_path, compressed_video_path)

        # Save session state
        st.session_state.result_path = compressed_video_path
        st.session_state.total_unique_pedestrians = total_unique_pedestrians
        st.session_state.processing = False

# Display result
if st.session_state.result_path:
    st.write("**Tracking and Counting Results**")
    st.video(st.session_state.result_path)
    st.write(f"**Total Unique Pedestrians**: {st.session_state.total_unique_pedestrians}")

# Reset app
if st.button("Reset and Upload New Video"):
    st.session_state.processing = False
    st.session_state.result_path = None
    st.session_state.total_unique_pedestrians = 0
    st.rerun()
