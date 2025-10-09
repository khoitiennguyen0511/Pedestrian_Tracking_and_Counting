# 👣 Pedestrians Tracking & Counting using YOLOv8 + DeepSORT

[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=python&logoColor=white)](https://ultralytics.com/)
[![DeepSORT](https://img.shields.io/badge/DeepSORT-4B8BBE?style=for-the-badge)](https://arxiv.org/abs/1703.07402)
[![PYTORCH](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

Hệ thống **phát hiện – theo dõi – đếm người đi bộ** theo thời gian thực, kết hợp **YOLOv8** (detect) và **DeepSORT** (tracking ID), có giao diện **Streamlit** trực quan. Chạy tốt trên Windows/macOS/Linux (CPU/GPU).

**Demo video:** [Demo](https://www.youtube.com/watch?v=cf1kmcOJJgg)

<p align="center">
  <img src="tracking_results/demo.gif" width="640" alt="Pedestrians Tracking Demo">
  <br><em>Hình ảnh theo dõi & đếm người đi bộ ở công viên</em>
</p>

---

## Mục lục
- [Tổng quan](#tổng-quan)
- [Tính năng](#tính-năng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Chạy trên Google Colab](#chạy-trên-google-colab)
- [Kết quả đánh giá](#kết-quả-đánh-giá)
- [Kết quả đếm & trực quan hoá](#kết-quả-đếm--trực-quan-hoá)
- [Cài đặt và chạy chương trình](#cài-đặt-và-chạy-chương-trình)
  - [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
  - [Cài đặt dependencies](#cài-đặt-dependencies)
  - [Chạy giao diện Streamlit](#chạy-giao-diện-streamlit)
  - [Chạy bằng script (CLI)](#chạy-bằng-script-cli)
- [Tuỳ chỉnh & mẹo hiệu năng](#tuỳ-chỉnh--mẹo-hiệu-năng)
- [Xử lý sự cố](#xử-lý-sự-cố)
- [Đóng góp](#đóng-góp)
- [Giấy phép & Ghi công](#giấy-phép--ghi-công)
- [Liên hệ](#liên-hệ)

---

## Tổng quan
Pipeline hoàn chỉnh để **phát hiện** người (`person`), **theo dõi ID ổn định** theo thời gian bằng **DeepSORT**, và **đếm** lượng người đi qua khung hình/khu vực. Ứng dụng hướng tới sự **dễ dùng** (Streamlit), **dễ mở rộng**, và **dễ triển khai**.

## Tính năng
- **Phát hiện pedestrians** bằng YOLOv8 (trọng số COCO hoặc fine-tune riêng).
- **Theo dõi đa đối tượng** với DeepSORT (gán ID nhất quán).
- **Đếm người** theo khung hình/line/zone (tuỳ logic trong app).
- **Giao diện Streamlit**: upload video, xem kết quả, lưu video đầu ra.
- **Hỗ trợ CPU/GPU** (CUDA nếu môi trường cho phép).
- **Log/biểu đồ** kết quả (khi bật lưu và trực quan hoá).

---

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
