import cv2
import numpy as np
import tkinter as tk
import pyautogui
from pynput.mouse import Controller,Button
from config import label_map
class ImgProcess:
    def __init__(self, image_shape):
        self.image_shape = image_shape
        self.width, self.height = image_shape[:2]
        self.center = (self.width // 2, self.height // 2)
        
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth() // 2
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        self.hand_data = None
        print("Screen size:", self.screen_width, self.screen_height)
        self.screen_point1 = (self.center[0] - 100, self.center[1] - 100)
        self.screen_point2 = (self.center[0] + 100, self.center[1] + 100)
        self.mouse_status = "release"  # 鼠标状态，初始为释放状态
        self.mouse_status_count = 0  # 鼠标状态计数器
        self.label_map = label_map
        
        self.mouse = Controller()
        
        # 平滑滤波参数
        self.ema_alpha = 0.2  # EMA平滑系数 (0.1-0.3效果最佳)
        self.prev_screen_x = self.screen_width // 2
        self.prev_screen_y = self.screen_height // 2
        
        # 移动阈值参数
        self.move_threshold = 3  # 像素移动阈值(忽略小于此值的移动)

    def draw_screen_point(self, image, p, text="1", color=(0, 255, 0)):
        cv2.circle(image, p, 5, color, -1)
        cv2.putText(image, text, (p[0] + 10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (188,188,188), 2)
        return image

    def draw_rect(self, image, p1, p2, color=(0, 255, 0)):
        cv2.rectangle(image, p1, p2, color, 2)
        return image
        
    def drag_screen_point(self, hand_data=None, gesture=None):
        if hand_data is not None and gesture is not None:
            p1 = hand_data.get(8, (0, 0))
            p2 = hand_data.get(4, (0, 0))
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            
            if gesture == "捏合 (Pinch)" and dist < 18:
                center_pos = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                dist2screen_point1 = np.linalg.norm(np.array(center_pos) - np.array(self.screen_point1))
                dist2screen_point2 = np.linalg.norm(np.array(center_pos) - np.array(self.screen_point2))
                mindist = min(dist2screen_point1, dist2screen_point2)
                
                if mindist < 45:
                    if dist2screen_point1 < dist2screen_point2:
                        self.screen_point1 = center_pos
                    else:
                        self.screen_point2 = center_pos
                    return True
        return False
    def gesture_group(self,hand_data):
        pass

    def control_mouse(self, hand_data=None, gesture=None):
        if hand_data is None or gesture is None :
            return
        
        point = hand_data.get(4, (0, 0))
        

        # 获取矩形的边界
        rect_left = min(self.screen_point1[0], self.screen_point2[0])
        rect_right = max(self.screen_point1[0], self.screen_point2[0])
        rect_top = min(self.screen_point1[1], self.screen_point2[1])
        rect_bottom = max(self.screen_point1[1], self.screen_point2[1])
        
        # 检查point是否在矩形区域内
        if (point[0] < rect_left - 20 or point[0] > rect_right + 20 or 
            point[1] < rect_top - 20 or point[1] > rect_bottom + 20 ):
            return
        
        # 计算归一化坐标
        x_ratio = (point[0] - rect_left) / (rect_right - rect_left) if rect_right != rect_left else 0
        y_ratio = (point[1] - rect_top) / (rect_bottom - rect_top) if rect_bottom != rect_top else 0
        
        # 映射到屏幕坐标
        screen_x = int(x_ratio * self.screen_width)
        screen_y = int(y_ratio * self.screen_height)
        
        # 计算移动距离
        move_distance = np.linalg.norm([screen_x - self.prev_screen_x, screen_y - self.prev_screen_y])
        
        # 应用移动阈值
        if move_distance < self.move_threshold:
            return
            
        # 应用指数移动平均(EMA)平滑
        smoothed_x = self.ema_alpha * screen_x + (1 - self.ema_alpha) * self.prev_screen_x
        smoothed_y = self.ema_alpha * screen_y + (1 - self.ema_alpha) * self.prev_screen_y
        
        # 更新鼠标位置
        if gesture == "握拳 (Fist)" or gesture == "一只食指 (Index)":
            self.mouse.position = (int(smoothed_x), int(smoothed_y))
            self.mouse.release(Button.left)
            self.mouse_status = "release"
            self.mouse_status_count = 0
            self.mouse.position = (int(smoothed_x), int(smoothed_y))
            self.prev_screen_x = smoothed_x
            self.prev_screen_y = smoothed_y



        elif gesture == "捏合 (Pinch)":
            self.mouse_status = "click&press"
            self.mouse_status_count += 1

            if self.mouse_status_count == 2:
                self.mouse.click(Button.left, 1)  # 单击左键
            elif self.mouse_status_count > 5:
                self.mouse.press(Button.left)

            self.mouse.position = (int(smoothed_x), int(smoothed_y))
            self.prev_screen_x = smoothed_x
            self.prev_screen_y = smoothed_y

                
        elif gesture == "食指和中指 (Index_Middle)":
            if self.mouse_status == "scroll&up":
                self.mouse_status_count += 1
            if self.mouse_status == "release":
                self.mouse.scroll(0, -1)
                self.mouse_status = "scroll&up"
                self.mouse_status_count += 1
            if self.mouse_status == "scroll&up" and self.mouse_status_count > 20:
                self.mouse.scroll(0, -1)


    def process_image(self, image, gesture=None, hand_data=None):
        self.hand_data = hand_data
        image = self.draw_screen_point(image, self.screen_point1, text="1", color=(0, 255, 0))
        image = self.draw_screen_point(image, self.screen_point2, text="2", color=(0, 255, 0))
        
        is_drag = self.drag_screen_point(hand_data, gesture)
        if not is_drag:
            self.control_mouse(hand_data, gesture)
        
        # 绘制矩形（确保左上角到右下角的顺序）
        rect_top_left = (min(self.screen_point1[0], self.screen_point2[0]), 
                         min(self.screen_point1[1], self.screen_point2[1]))
        rect_bottom_right = (max(self.screen_point1[0], self.screen_point2[0]), 
                             max(self.screen_point1[1], self.screen_point2[1]))
        image = self.draw_rect(image, rect_top_left, rect_bottom_right, color=(0, 0, 255))
        
        return image
