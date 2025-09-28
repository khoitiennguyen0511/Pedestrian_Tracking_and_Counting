
# 🧍 Pedestrian Tracking and Counting using YOLOv8 + DeepSORT

This project implements a real-time pedestrian tracking and counting system using **YOLOv8** for object detection and **DeepSORT** for multi-object tracking. The goal is to count the number of people crossing a virtual line in a video stream (from webcam or file).

> 🔍 Built as a practical application of Computer Vision in public safety, smart cities, and retail analytics.

---

## 📌 Key Features

- ✅ Real-time pedestrian detection with **YOLOv8**
- ✅ Multi-object tracking using **DeepSORT** (appearance + motion)
- ✅ Accurate **people counting** based on line crossing logic
- ✅ Input support: **video files**, **webcam**, **RTSP streams**
- ✅ Output annotated video with **bounding boxes, IDs, counters**

---

## 🎥 Demo

<p align="center">
  <img src="https://github.com/khoitiennguyen0511/Pedestrian_Tracking_and_Counting/assets/demo.gif" width="720" />
</p>

# Pedestrian Tracking and Counting

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pedestriantrackingandcounting.streamlit.app/)


---

## 🧠 Model Architecture

| Component   | Description |
|-------------|-------------|
| **YOLOv8**  | State-of-the-art object detector for person class |
| **DeepSORT** | Tracker combining appearance embeddings + Kalman Filter |
| **Counting Logic** | Triggered when an object's center crosses a virtual line |

---

## 🗂️ Project Structure

