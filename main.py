import cv2
from src.hand_detector import HandDetector
from src.gesture_predictor import GesturePredictor
from src.img_process import ImgProcess
from src.tools.data_utils import convert_landmarks_to_list
"""
1.使用opencv来获取摄像头信息
2.使用mediapipe将手的信息传给opencv
"""
url = "http://192.168.0.122:8080"
SIZE = (640, 480)
#打开电脑上的摄像头,笔记本的参数是0
camera = cv2.VideoCapture(0)
#在另一个文件中创建一个类，里面存放mediapipe实现的功能，并导入该文件
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
        #获取到左手的食指指尖，如果获取不到返回None
        left_finger = position['Left'].get(8,None)
        left_bigger = position['Left'].get(4,None)
        # print(left_finger)
        # if left_finger:
        #     cv2.circle(img,(left_finger[0],left_finger[1]),10,(0,0,255),cv2.FILLED)
        # if left_bigger:
        #     cv2.circle(img, (left_bigger[0], left_bigger[1]), 10, (255, 0, 0), cv2.FILLED)

        
        right_bigger = position['Right'].get(4,None)
        right_finger = position['Right'].get(8, None)
        
        if len(position['Right']) >= 10:
            # print(position['Right'])
            # draw_rect(img, (right_bigger[0], right_bigger[1]), (right_finger[0], right_finger[1]), color=(255, 0, 0))
            res = convert_landmarks_to_list(position['Right'])
            predictions = gesture_detector.predict(res)
            print("predictions:", predictions)

        img = drawer.process_image(img, predictions, position['Right'])


        cv2.imshow('Video',img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
 
#关闭摄像头，解除程序占用摄像头
camera.release()
#cv2把所有打开的窗口关闭掉
cv2.destroyAllWindows()