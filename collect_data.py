import cv2
from src.hand_detector import HandDetector
from src.gesture_predictor import GesturePredictor
from src.tools.data_utils import * 
from src.img_process import ImgProcess



# 0: '捏合 (Pich)', 1: '握拳 (Fist)', 2: '手掌 (Palm)', 3: '食指和中指 (Index_Middle)', 4: '一只食指 (Index)'
data_lable = 0

SIZE = (640, 480)
camera = cv2.VideoCapture(0)
hand_detector = HandDetector()
gesture_detector = GesturePredictor("model/hand_gesture_model.pth")
drawer = ImgProcess(SIZE)
count = 0
predictions = None
while True:
    count += 1
    #从摄像头里面读,获取摄像头视频帧的数据
    #会返回两个数据，一个是判断是否读成功，另一个是返回视频帧的图片
    success,img = camera.read()
    #如果读取成功
    if success:
        #opencv获取到的视频是翻转的，需要将视频改为镜像的
        img = cv2.flip(img,1)
        img = cv2.resize(img, (SIZE))
        #调用类里面的函数process,将img放到mediapipe中进行处理
        hand_detector.process(img,False)
        #储存position字典
        position = hand_detector.find_position(img)
        
        if len(position['Right']) >= 10:
            res = convert_landmarks_to_list(position['Right'])
            save_to_csv(position['Right'], 'Right', data_lable, filename="data/data.csv")

        cv2.imshow('Video', img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
 
#关闭摄像头，解除程序占用摄像头
camera.release()
#cv2把所有打开的窗口关闭掉
cv2.destroyAllWindows()