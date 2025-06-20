import cv2
import time
def list_available_cameras():
    """
    列出可用的摄像头
    """
    available_cameras = []
    for i in range(10):  # 通常尝试0-9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def test_camera(camera_index):
    """
    测试特定摄像头是否可用
    """
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        # 尝试读取一帧
        ret, frame = cap.read()
        cap.release()
        return ret
    return False

# 获取可用摄像头列表
cameras = list_available_cameras()
print("可用摄像头索引:", cameras)

# # 打开第一个可用摄像头
# if cameras:
#     cap = cv2.VideoCapture(cameras[0])
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         cv2.imshow('Camera', frame)
        
#         # 按 'q' 键退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()
# else:
#     print("未找到可用摄像头")