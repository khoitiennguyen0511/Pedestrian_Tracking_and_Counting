# Pedestrians Tracking & Counting using YOLOv8 + DeepSORT on Streamlit

[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=python&logoColor=white)](https://ultralytics.com/)
[![DeepSORT](https://img.shields.io/badge/DeepSORT-4B8BBE?style=for-the-badge)](https://arxiv.org/abs/1703.07402)
[![PYTORCH](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

Hệ thống **phát hiện – theo dõi – đếm người đi bộ** theo thời gian thực, kết hợp **YOLOv8** (detect) và **DeepSORT** (tracking ID), có giao diện **Streamlit** trực quan. Chạy tốt trên Windows/macOS/Linux (CPU/GPU).

**Link video:** [Demo](https://www.youtube.com/watch?v=cf1kmcOJJgg)

<p align="center">
  <img src="https://github.com/khoitiennguyen0511/Pedestrian_Tracking_and_Counting/blob/main/assets/pedestrian_demo.gif" width="640" alt="Pedestrians Tracking Demo">
  <br><em>Hình ảnh theo dõi & đếm người đi bộ ở công viên</em>
</p>

---

## Pipeline project
<p align="center">
  <img src="https://github.com/khoitiennguyen0511/Pedestrian_Tracking_and_Counting/blob/main/assets/pipelone.jpg" width="640" alt="Pedestrians Tracking pipeline">
  <br><em>Pipeline project</em>
</p>

## Mục lục
- [Tổng quan](#tổng-quan)
- [Tính năng](#tính-năng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Chạy trên Google Colab](#chạy-trên-google-colab)
- [Kết quả đánh giá](#kết-quả-đánh-giá)
- [Kết quả đếm](#kết-quả-đếm)
- [Cài đặt và chạy chương trình](#cài-đặt-và-chạy-chương-trình)
  - [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
  - [Cài đặt dependencies](#cài-đặt-dependencies)
  - [Chạy giao diện Streamlit](#chạy-giao-diện-streamlit)
- [Liên hệ](#liên-hệ)

## Tổng quan
Pipeline hoàn chỉnh để **phát hiện** người, **theo dõi ID ổn định** theo thời gian bằng **DeepSORT**, và đếm lượng người đi qua khung hình/khu vực. Ứng dụng hướng tới sự dễ dùng (Streamlit), dễ mở rộng, và dễ triển khai.

## Tính năng
- **Phát hiện pedestrians** bằng YOLOv8 (fine-tune trên MOT17).
- **Theo dõi đa đối tượng** với DeepSORT (gán ID nhất quán).
- **Đếm người** theo khung hình/line/zone (tuỳ logic trong app).
- **Giao diện Streamlit**: upload video, xem kết quả, lưu video đầu ra.
- **Hỗ trợ CPU/GPU** (CUDA nếu môi trường cho phép).

## Cấu trúc dự án
```text
Pedestrian_Tracking_and_Counting/
├─ deep_sort/                            
├─ resources/
│  └─ networks/                          
├─ assets/
├─ runs/
│  └─ detect/                            # Kết quả Inference
├─ tracking_results/                     # Video kết quả xuất ra
├─ weights/                              
├─ .devcontainer/  .vscode/              
├─ Object_Detection.ipynb                
├─ YOLOv8_DeepSORT_Tracking_and_Counting.ipynb
├─ app.py                                # Deploy Streamlit
├─ yolov8_deepsort.py                    
├─ requirements.txt
├─ utils.py
├─ video.mp4                             # Video mẫu
└─ README.md
```
> **Lưu ý:** DeepSORT cần **embedding model** (ví dụ: `resources/networks/mars-small128.pb`). Hãy giữ **đúng đường dẫn** trong code.
---

## Chạy trên Google Colab
- **Detect (YOLOv8):**  [Object_Detection.ipynb](https://colab.research.google.com/drive/1xRHFuxw0sm5G6vp3cQZxTtNwN3ego55g#scrollTo=_JYXSm8QYx2U)
- **Tracking (DeepTRACK):**  [YOLOv8_DeepSORT_Tracking.ipynb](https://colab.research.google.com/drive/1IfCSu1GW6ioWrZOI2KidIVk9MZ-flQS6)

## Kết quả đánh giá

| Metric        | Value |
|:--------------|-----:|
| Precision (P) | 0.902  |
| Recall (R)    | 0.815 |
| mAP@0.50      | 0.908 |
| mAP@0.50:0.95 | 0.65 |

## Kết quả đếm

<p align="center">
  <img src="https://github.com/khoitiennguyen0511/Pedestrian_Tracking_and_Counting/blob/main/assets/pedestrian_counting.jpg" alt="Biểu đồ đếm pedestrians" width="800">
  <br><em>Hình ảnh kết quả người đi bộ ở công viên</em>
</p>

---

## Cài đặt và chạy chương trình

### Yêu cầu hệ thống
- **OS:** Windows 10/11, Ubuntu 20.04+, macOS 12+  
- **Python:** 3.8+
- **FFmpeg:** để đọc/ghi video  
- **(Tuỳ chọn) GPU:** NVIDIA CUDA + PyTorch CUDA tương thích

### Cài đặt dependencies

```bash
# 1) Tạo môi trường ảo
python -m venv pedestrian_env

# Windows
pedestrian_env\Scripts\activate
# macOS/Linux
# source pedestrian_env/bin/activate

# 2) Clone repo & cài gói
git clone https://github.com/khoitiennguyen0511/Pedestrian_Tracking_and_Counting.git
cd Pedestrian_Tracking_and_Counting
pip install -r requirements.txt

# 3) Cài FFmpeg
# Windows (winget)
winget install ffmpeg
# Ubuntu/Debian
# sudo apt update && sudo apt install -y ffmpeg
# macOS (Homebrew)
# brew install ffmpeg
```

### Chạy giao diện Streamlit
```
streamlit run app.py
# hoặc:
# python -m streamlit run app.py
```
> Chọn video.mp4 trong repo hoặc upload video định dạng .mp4, .avi, .webm, .mpeg4 (mặc định giới hạn **~200MB**).

## Liên hệ
- GitHub: [@khoitiennguyen0511](https://github.com/khoitiennguyen0511)
- Email: khoitiennguyen2004l@gmail.com
- Linkln: [Tiến Khôi Nguyễn](https://www.linkedin.com/in/ti%E1%BA%BFn-kh%C3%B4i-nguy%E1%BB%85n-63732935b/)

⭐ Nếu bạn thấy dự án hữu ích, đừng quên cho repository một star!
