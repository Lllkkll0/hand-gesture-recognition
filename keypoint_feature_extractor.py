import numpy as np
import cv2
import mediapipe as mp

class HandFeatureExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_keypoint_features(self, image):
        """提取手部关键点特征"""
        # 将BGR转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        results = self.hands.process(image_rgb)
        
        features = []
        hand_present = False
        
        if results.multi_hand_landmarks:
            hand_present = True
            
            for hand_landmarks in results.multi_hand_landmarks:
                # 提取所有21个关键点的坐标
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                
                # 关键点特征
                landmarks = np.array(landmarks).flatten()
                features.extend(landmarks)

                # 计算指尖与手掌的距离（高级特征）
                palm_center = np.mean([
                    [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y],
                    [hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y],
                    [hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y]
                ], axis=0)
                # 指尖索引
                fingertips = [4, 8, 12, 16, 20]  # 拇指、食指、中指、无名指、小指
                
                # 计算每个指尖到手掌中心的距离
                for tip_idx in fingertips:
                    tip = [hand_landmarks.landmark[tip_idx].x, hand_landmarks.landmark[tip_idx].y]
                    distance = np.sqrt((tip[0] - palm_center[0])**2 + (tip[1] - palm_center[1])**2)
                    features.append(distance)
                
                # 计算指尖之间的角度
                angles = []
                for i in range(len(fingertips)):
                    for j in range(i+1, len(fingertips)):
                        tip1 = [hand_landmarks.landmark[fingertips[i]].x, hand_landmarks.landmark[fingertips[i]].y]
                        tip2 = [hand_landmarks.landmark[fingertips[j]].x, hand_landmarks.landmark[fingertips[j]].y]
                        
                        # 计算向量
                        v1 = np.array(tip1) - np.array(palm_center)
                        v2 = np.array(tip2) - np.array(palm_center)
                        
                        # 计算角度
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                        angles.append(angle)
                
                features.extend(angles)
        
        features = np.array(features).flatten()
        
        return features, hand_present