import cv2
import numpy as np
import os
import streamlit as st
from ultralytics import YOLO
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define Detector function
class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, src_img):
        results = self.model.predict(src_img, save=False)[0]
        bboxes = results.boxes.xywh.cpu().numpy()
        bboxes[:, :2] = bboxes[:, :2] - (bboxes[:, 2:] / 2)
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        return bboxes, scores, class_ids

# Define DeepSORT function
class DeepSORT:
    def __init__(self,
                 model_path='resources/networks/mars-small128.pb',
                 max_cosine_distance=0.7,
                 nn_budget=None,
                 classes=['object']):
        self.encoder = gdet.create_box_encoder(model_path, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metric)

        key_list = []
        val_list = []
        for ID, class_name in enumerate(classes):
            key_list.append(ID)
            val_list.append(class_name)

        self.key_list = key_list
        self.val_list = val_list

    def tracking(self, origin_frame, bboxes, scores, class_ids):
        features = self.encoder(origin_frame, bboxes)
        detections = [
            Detection(bbox, score, class_id, feature)
            for bbox, score, class_id, feature in
            zip(bboxes, scores, class_ids, features)
        ]

        self.tracker.predict()
        self.tracker.update(detections)

        tracked_bboxes = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
            bbox = track.to_tlbr()
            class_id = track.get_class()
            conf_score = track.get_conf_score()
            tracking_id = track.track_id
            tracked_bboxes.append(bbox.tolist() + [class_id, conf_score, tracking_id])

        return np.array(tracked_bboxes)

# Define draw_detection function
def draw_detection(img, bboxes, scores, class_ids, ids,
                   classes=['objects'], mask_alpha=0.3):
    height, width = img.shape[:2]
    np.random.seed(0)
    rng = np.random.default_rng(3)
    colors = rng.uniform(0, 255, size=(len(classes), 3))

    mask_img = img.copy()
    det_img = img.copy()

    size = min([height, width]) * 0.0006
    text_thickness = int(min([height, width]) * 0.001)

    for bbox, score, class_id, id_ in zip(bboxes, scores, class_ids, ids):
        color = colors[class_id]
        x1, y1, x2, y2 = bbox.astype(int)

        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        label = classes[class_id]
        caption = f'{label} {int(score * 100)}% ID: {id_}'
        (tw, th), _ = cv2.getTextSize(text=caption,
                                      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size,
                                      thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
        cv2.putText(det_img, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, size,
                    (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

# Video tracking for Streamlit
def video_tracking(video_path, detector, tracker, is_save_result=False, save_dir='tracking_results'):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if is_save_result:
        os.makedirs(save_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_result_name = 'output_video.avi'
        save_result_path = os.path.join(save_dir, save_result_name)
        out = cv2.VideoWriter(save_result_path, fourcc, fps, (width, height))
    else:
        save_result_path = None
        out = None

    all_tracking_results = []
    tracked_ids = set()
    frame_count = 0
    stframe = st.empty()  # Streamlit placeholder

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        bboxes, scores, class_ids = detector.detect(frame)

        tracker_pred = tracker.tracking(frame, bboxes, scores, class_ids)

        current_count = 0
        total_unique_count = len(tracked_ids)

        if tracker_pred.size > 0:
            bboxes = tracker_pred[:, :4]
            class_ids = tracker_pred[:, 4].astype(int)
            conf_scores = tracker_pred[:, 5]
            tracking_ids = tracker_pred[:, 6].astype(int)

            new_ids = set(tracking_ids) - tracked_ids
            tracked_ids.update(new_ids)

            current_count = len(set(tracking_ids))
            total_unique_count = len(tracked_ids)

            result_img = draw_detection(frame, bboxes, conf_scores, class_ids, tracking_ids)
        else:
            result_img = frame

        cv2.putText(result_img, f'Frame: {frame_count}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
        cv2.putText(result_img, f'Current Count: {current_count}', (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
        cv2.putText(result_img, f'Total Unique: {total_unique_count}', (10, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)

        all_tracking_results.append(tracker_pred)

        if is_save_result:
            out.write(result_img)

        stframe.image(result_img, channels="BGR")

    cap.release()
    if is_save_result:
        out.release()

    return all_tracking_results, len(tracked_ids), save_result_path
