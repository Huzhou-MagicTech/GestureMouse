import torch
import torch.nn as nn
import numpy as np

class GesturePredictor:
    """
    一个用于加载预训练手势识别模型并进行预测的类。
    """
    
    # --- 1. 将模型定义嵌套在类中 ---
    # 这使得类更加自包含，因为模型的结构是加载权重所必需的。
    class NeuralNet(nn.Module):
        def __init__(self, input_size, num_classes):
            super(GesturePredictor.NeuralNet, self).__init__()
            self.layer1 = nn.Linear(input_size, 128)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(128, 64)
            self.layer3 = nn.Linear(64, num_classes)

        def forward(self, x):
            out = self.layer1(x)
            out = self.relu(out)
            out = self.layer2(out)
            out = self.relu(out)
            out = self.layer3(out)
            return out

    # --- 2. 初始化Predictor ---
    def __init__(self, model_path):
        """
        初始化手势预测器。
        Args:
            model_path (str): 已训练模型 (.pth) 文件的路径。
        """
        # 定义设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 定义模型参数 (必须与训练时完全一致)
        self.input_size = 21 * 3
        self.num_classes = 5
        
        # 实例化模型结构
        self.model = self.NeuralNet(self.input_size, self.num_classes).to(self.device)
        
        # 加载训练好的权重
        # map_location 确保了即使在没有GPU的设备上也能加载在GPU上训练的模型
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # 将模型设置为评估模式
        # 这是非常重要的一步，它会关闭dropout和batch normalization等训练特有的层
        self.model.eval()

        # 定义标签映射，用于将输出索引转换为可读的标签
        self.label_map = {
            0: '捏合 (Pinch)', 
            1: '握拳 (Fist)', 
            2: '手掌 (Palm)', 
            3: '食指和中指 (Index_Middle)', 
            4: '一只食指 (Index)'
        }
        print(f"模型已从 '{model_path}' 加载，正在使用 {self.device} 设备。")

    # --- 3. 定义预测方法 ---
    def predict(self, landmarks_list):
        """
        对单次手部关键点数据进行预测。
        
        Args:
            landmarks_list (list): 一个包含63个浮点数 (21 * 3) 的扁平列表。
        
        Returns:
            str: 预测出的手势名称。
        """
        # 将列表转换为 PyTorch 张量
        input_tensor = torch.tensor(landmarks_list, dtype=torch.float32).to(self.device)
        
        # --- FIX ---
        # 将张量reshape为 [1, 63]，以匹配模型的输入要求
        # .view(1, -1) 会自动推断维度，将输入转换为一个批次大小为1的扁平向量
        input_tensor = input_tensor.view(1, -1)

        # 之前的 .unsqueeze(0) 就不再需要了，因为 .view(1, -1) 已经包含了批次维度

        # 使用 `torch.no_grad()` 上下文管理器进行推理
        with torch.no_grad():
            # 获取模型的原始输出 (logits)
            outputs = self.model(input_tensor)
            
            # 使用 argmax 获取概率最高的类别的索引
            predicted_idx = torch.argmax(outputs, dim=1).item()

        # 将索引映射回人类可读的标签
        return self.label_map.get(predicted_idx, "未知类别")

# --- 如何使用这个类 ---
if __name__ == '__main__':
    # 这是一个示例，展示了如何使用 HandPosePredictor 类。
    # 在你的实际应用中，`res` 会来自 MediaPipe 的实时视频流。
    
    # 假设你的模型文件叫 "hand_gesture_model.pth" 并且在同一个目录下
    model_file_path = 'hand_gesture_model.pth'

    # 1. 实例化预测器
    # 这会在初始化时加载模型，只需要执行一次。
    try:
        gesture_detector = HandPosePredictor(model_file_path)
    except FileNotFoundError:
        print(f"错误: 模型文件 '{model_file_path}' 未找到。")
        print("请确保模型文件与此脚本位于同一目录，或提供正确的文件路径。")
        exit()

    # 2. 准备一个模拟的输入数据
    # 在你的实际代码中，`hand_landmarks_sample` 会是 mediapipe 的输出结果，
    # 比如 `results.hand_landmarks[0]`。
    # 这里我们用随机数据创建一个结构相同的模拟对象。
    class Landmark:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class HandLandmarks:
        def __init__(self, landmarks):
            self.landmark = landmarks

    # 创建21个随机关键点
    mock_landmarks = [Landmark(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(21)]
    hand_landmarks_sample = HandLandmarks(mock_landmarks)


    # 3. 进行预测
    # 你可以在一个循环中（例如，处理视频帧）反复调用此方法。
    prediction = gesture_detector.predict(hand_landmarks_sample)

    # 4. 打印结果
    print(f"\n模拟输入数据的预测结果是: {prediction}")

    # 示例：处理没有检测到手的情况
    no_hand_prediction = gesture_detector.predict(None)
    print(f"未检测到手时的预测结果是: {no_hand_prediction}")
