# Pedestrians Tracking and Counting using YOLOv8 + DeepSORT

# Steps to run code

Step 1: Install virtual environment

```bash
python -m venv pedestrian_env
```

- On Window:

```bash
pedestrian_env\Scripts\activate
```

- On MacOs/Linux:

```bash
source pedestrian_env/bin/activate
```

Once enabled, you will see a virtual environment at the beginning of the command line, for example:

```bash
(pedestrian_env) PS C:\Users\tienk> 
```

Step 2: Clone the repository and install the dependencies

```
git clone https://github.com/khoitiennguyen0511/Pedestrian_Tracking_and_Counting.git
```

- Goto the cloned folder.

```bash
cd Pedestrian_Tracking_and_Counting
```

```bash
pip install -r requirements.txt
```

```bash
winget install ffmpeg
```

Step 3: Run code

```bash
streamlit run app.py
```

This, a window appears and you choose a video file (MP4, AVI, WEBM, MEPG4). I recommend to choose video.mp4 file in repo.
