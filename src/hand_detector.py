import cv2
import mediapipe as mp
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class HandDetector():
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 smoothing_factor=0.45):  # 新增平滑参数
        # 配置 GPU 选项
        mp_hands_options = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.hand_detector = mp_hands_options
        self.drawer = mp.solutions.drawing_utils
        self.smoothing_factor = smoothing_factor  # 平滑系数 (0-1)
        self.prev_positions = {}  # 存储上一帧的位置

    def process(self, img, draw=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.hand_data = self.hand_detector.process(img_rgb)
        # if draw:
        #     if self.hand_data.multi_hand_landmarks:
        #         for handlms in self.hand_data.multi_hand_landmarks:
        #             self.drawer.draw_landmarks(img, handlms, mp.solutions.hands.HAND_CONNECTIONS)
        return img

    def _smooth_position(self, current_positions):
        """使用指数移动平均平滑当前位置（包括深度信息）"""
        smoothed_positions = {'Left': {}, 'Right': {}}
        
        for hand_type in ['Left', 'Right']:
            if hand_type in current_positions:
                for id, pos in current_positions[hand_type].items():
                    # 每个位置是 (x, y, z) 三元组
                    x, y, z = pos
                    
                    # 如果存在历史位置，则进行平滑
                    if hand_type in self.prev_positions and id in self.prev_positions[hand_type]:
                        prev_x, prev_y, prev_z = self.prev_positions[hand_type][id]
                        new_x = int(prev_x * (1 - self.smoothing_factor) + x * self.smoothing_factor)
                        new_y = int(prev_y * (1 - self.smoothing_factor) + y * self.smoothing_factor)
                        new_z = prev_z * (1 - self.smoothing_factor) + z * self.smoothing_factor
                        smoothed_positions[hand_type][id] = (new_x, new_y, new_z)
                    else:
                        # 没有历史数据，直接使用当前值
                        smoothed_positions[hand_type][id] = (x, y, z)
        
        # 更新历史位置
        self.prev_positions = smoothed_positions.copy()
        return smoothed_positions

    def find_position(self, img, draw=True):
        h, w, c = img.shape
        position = {'Left':{}, 'Right':{}}
        if self.hand_data.multi_hand_landmarks:
            i = 0
            for point in self.hand_data.multi_handedness:
                score = point.classification[0].score
                if score >= 0.8:
                    label = point.classification[0].label
                    hand_lms = self.hand_data.multi_hand_landmarks[i].landmark
                    
                    for id, lm in enumerate(hand_lms):
                        x, y = int(lm.x * w), int(lm.y * h)
                        z = lm.z  # 获取深度信息
                        position[label][id] = (x, y, z)
                i += 1
        
        # 应用平滑滤波（现在包含深度信息）
        smoothed_positions = self._smooth_position(position)
        
        # 可选：在图像上绘制平滑后的点
        if draw:
            for hand_type, points in smoothed_positions.items():
                for id, (x, y, z) in points.items():
                    if id == 8 or id == 4 :
                        cv2.circle(img, (x, y), 5, (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, f"{id}", (x + 10, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        # 在点旁边显示深度值
                        cv2.putText(img, f"z:{z:.2f}", (x + 10, y + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        return smoothed_positions
    


    