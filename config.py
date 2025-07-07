input_size = 21 * 3  # 21个3D坐标点 (x, y, z)
learning_rate = 0.001
batch_size = 16
num_epochs = 20      # 训练轮次，你可以根据需要调整
model_save_path = 'model/hand_gesture_model.pth' # 定义模型保存路径


label_map = {
            0: '捏合 (Pinch)', 
            1: '握拳 (Fist)', 
            2: '手掌 (Palm)', 
            3: '食指和中指 (Index_Middle)', 
            4: '一只食指 (Index)',
            # 添加示例
            # 5: 'OK手势 (OK)',
            # 6: '中指 (Middle)',
        }

num_classes = len(label_map) 