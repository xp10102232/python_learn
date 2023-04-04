import numpy as np
import cv2
import time

class knnDetector:
    def __init__(self, history, dist2Threshold, minArea):
        self.minArea = minArea
        self.detector = cv2.createBackgroundSubtractorKNN(history, dist2Threshold, False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detectOneFrame(self, frame):
        if frame is None:
            return None
        # 前景分离
        mask = self.detector.apply(frame)
        
        # 形态学滤波
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel)
        
        # 轮廓检测
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
        i = 0
        bboxs = []

        # 去掉面积过小的轮廓        
        for c in contours:
            i += 1
            if cv2.contourArea(c) < self.minArea:
                continue
            # 获取外接矩形
            bboxs.append(cv2.boundingRect(c))

        return mask, bboxs





        




