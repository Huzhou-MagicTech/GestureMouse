import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os # 导入 os 模块来处理文件路径
from config import * 


# Device configuration (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 2. 创建自定义数据集 ---
class HandGestureDataset(Dataset):
    """自定义手势数据集"""
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): 带注释的CSV文件的路径。
        """
        self.data_frame = pd.read_csv(csv_file)
        # 最后一列是标签 'label'
        self.labels = self.data_frame.iloc[:, -1].values
        # 除了最后两列 ('name', 'label') 外，其余都是特征
        self.features = self.data_frame.iloc[:, :-2].values

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 将数据转换为PyTorch张量
        features = torch.tensor(self.features[idx, :], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

# --- 3. 定义神经网络模型 ---
class NeuralNet(nn.Module):
    """一个简单的前馈神经网络"""
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
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

# --- 4. 初始化数据集和数据加载器 ---
try:
    dataset = HandGestureDataset(csv_file='data/data.csv')
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True)
except FileNotFoundError:
    print("错误：'data/data.csv' 未找到。")
    print("请确保你的项目结构正确，并且CSV文件存在。")
    exit()

# --- 5. 初始化模型、损失函数和优化器 ---
model = NeuralNet(input_size=input_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 6. 训练模型 ---
print("开始训练...")
total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0: # 调整打印频率
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

print("训练完成!")

# --- 7. 保存模型到本地 ---
# 使用 torch.save() 保存模型的 state_dict
torch.save(model.state_dict(), model_save_path)
print(f"模型已成功保存到: {model_save_path}")

# --- 8. 加载模型并进行预测的示例 ---
print("\n--- 加载模型并进行预测的示例 ---")

# 检查模型文件是否存在
if os.path.exists(model_save_path):
    # (1) 首先，需要重新实例化一个同样结构的模型
    # 注意：这里我们创建了一个新的模型实例 `loaded_model`
    loaded_model = NeuralNet(input_size=input_size, num_classes=num_classes).to(device)

    # (2) 然后，将保存的 state_dict 加载到新模型中
    loaded_model.load_state_dict(torch.load(model_save_path))

    # (3) 必须调用 model.eval() 将模型设置为评估模式
    # 这会关闭 Dropout 和 BatchNorm 等层，在预测时很重要
    loaded_model.eval()

    print("模型已成功加载。")

    # (4) 使用加载的模型进行预测
    # 我们用数据集中的第一条数据作为示例
    sample_features, sample_label = dataset[0]
    # 需要增加一个 batch 维度 (unsqueeze(0))，因为模型期望的是批次数据
    sample_features = sample_features.to(device).unsqueeze(0)

    # 在 `with torch.no_grad()` 上下文中进行预测，可以节省内存并加速
    with torch.no_grad():
        prediction = loaded_model(sample_features)
        # 使用 argmax 获取最高分数的类别索引
        predicted_class = torch.argmax(prediction, dim=1).item()

    # 定义标签映射以显示人类可读的名称

    print(f"\n测试样本的真实标签: {label_map[sample_label.item()]}")
    print(f"加载的模型预测结果: {label_map[predicted_class]}")

else:
    print(f"模型文件 {model_save_path} 不存在。")