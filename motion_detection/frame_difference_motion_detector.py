import numpy as np
import cv2

class FrameDifferenceMotionDetector:
    def __init__(self, threshold=30, min_area=500):
        self.prev_frame = None
        self.threshold = threshold
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> dict:
        result = {
            'motion_mask': None,
            'motion_bbox': [],
            'motion_contours': [],
            'motion_vectors': []
        }
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.prev_frame is None:
            self.prev_frame = gray
            return result
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        filtered_contours = []
        motion_vectors = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            bboxes.append((x, y, w, h))
            filtered_contours.append(c.reshape(-1, 2).tolist())
            prev_roi = self.prev_frame[y:y+h, x:x+w]
            curr_roi = gray[y:y+h, x:x+w]
            flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mean_flow = flow.reshape(-1, 2).mean(axis=0)
            center = (int(x + w/2), int(y + h/2))
            vector = (float(mean_flow[0]), float(mean_flow[1]))
            motion_vectors.append({'center': center, 'vector': vector})
        result['motion_mask'] = thresh
        result['motion_bbox'] = bboxes
        result['motion_contours'] = filtered_contours
        result['motion_vectors'] = motion_vectors
        self.prev_frame = gray
        return result
