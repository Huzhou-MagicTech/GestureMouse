# GestureMouse 项目文档

## 一、项目概述
GestureMouse 是一个基于手势识别的鼠标控制项目，利用计算机视觉和深度学习技术，通过摄像头捕捉手部动作，识别不同的手势并将其映射为鼠标操作，实现无需物理鼠标的交互体验。
## 二、项目结构
```
GestureMouse/
├── Makefile
├── collect_data.py
├── main.py
├── train.py
├── src/
│   ├── hand_detector.py
│   ├── gesture_predictor.py
│   ├── img_process.py
│   └── tools/
│       ├── data_utils.py
│       └── get_capture.py
├── data/
│   └── data.csv
└── model/
    └── hand_gesture_model.pth
```


### 主要文件及功能说明
1. Makefile：定义了项目的常用命令，如运行程序、训练模型、收集数据等。
2. collect_data.py：用于收集手部关键点数据，并将其保存到 CSV 文件中，为模型训练做准备。
3. main.py：项目的主程序，负责从摄像头读取视频流，进行手势识别和鼠标控制。
4. train.py：使用收集到的数据训练手势识别模型，并将训练好的模型保存到本地。
5. src/hand_detector.py：使用 MediaPipe 库检测手部关键点，并对关键点数据进行平滑处理。
6. src/gesture_predictor.py：加载预训练的手势识别模型，对输入的手部关键点数据进行预测。
7. src/img_process.py：处理视频帧，绘制相关信息，并根据手势控制鼠标操作。
8. src/tools/data_utils.py：提供数据处理工具，如将关键点数据保存到 CSV 文件、将关键点字典转换为列表或数组等。
9. src/tools/get_capture.py：列出可用的摄像头，并测试特定摄像头是否可用。
10. data/data.csv：存储收集到的手部关键点数据。
11. model/hand_gesture_model.pth：保存训练好的手势识别模型。

## 三、环境搭建
依赖安装
在项目根目录下，使用以下命令安装所需的 Python 依赖：

```bash
pip install -r requirements.txt
```

依赖库如下
```
torch
torchvision
opencv-python
mediapipe
pandas
numpy
tkinter
pyautogui
pynput
```

## 四、使用方法

### 1. 收集数据
在项目根目录下，运行以下命令收集手部关键点数据：
```bash
make collect
```
0: '捏合 (Pich)', 1: '握拳 (Fist)', 2: '手掌 (Palm)', 3: '食指和中指 (Index_Middle)', 4: '一只食指 (Index)'
请根据收集的标签修改 data_lable，使其对应你的手势
运行该命令后，摄像头将打开，你可以做出不同的手势，程序会自动将手部关键点数据保存到 data/data.csv 文件中。按 q 键退出数据收集过程。（文件形式是以追加模式打开）

### 2. 训练模型
训练好模型后，运行以下命令启动手势控制鼠标程序：
```bash
make run
```
运行该命令后，摄像头将打开，程序会实时识别你的手势并控制鼠标操作。按 q 键退出程序。


## 五、手势映射

目前预训练模型支持以下 5 种手势，并将其映射为相应的鼠标操作，其余的根据自己需求修改：
捏合 (Pinch)：鼠标左键单击并按住
握拳 (Fist)：鼠标移动
手掌 (Palm)：无特定操作
食指和中指 (Index_Middle)：鼠标向上滚动
一只食指 (Index)：鼠标移动
