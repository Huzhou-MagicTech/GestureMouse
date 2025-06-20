import numpy as np
import csv
import os

def save_to_csv(data, position, label, filename="keypoint_data.csv"):
    """
    将关键点数据保存到CSV文件中
    
    参数:
    data: 字典格式的关键点数据 {id: (x, y, z)}
    position: 位置字符串 (如'Right')
    label: 标签值 (整数)
    filename: CSV文件名
    """
    # 准备表头和数据行
    headers = []
    row_data = []
    
    # 按ID顺序处理每个点 (0到20)
    for point_id in sorted(data.keys()):
        x, y, score = data[point_id]
        # 添加表头
        headers.extend([f"x{point_id}", f"y{point_id}", f"z{point_id}"])
        # 添加数据
        row_data.extend([x, y, score])
    
    # 添加位置和标签
    headers.extend(["name", "label"])
    row_data.extend([position, label])
    
    # 检查文件是否存在，决定是否写入表头
    file_exists = os.path.isfile(filename)
    
    # 写入数据
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow(headers)
        
        # 写入数据行
        writer.writerow(row_data)
    # print(f"数据已保存到 {filename}")



def convert_landmarks_to_list(landmarks_dict):
    """
    将关键点字典转换为列表形式
    
    Args:
        landmarks_dict (dict): 关键点字典
    
    Returns:
        list: 关键点坐标列表 [[x1,y1,z1], [x2,y2,z2], ...]
    """
    # 方法1：按字典顺序提取坐标
    landmarks_list = [list(landmarks_dict[i]) for i in range(len(landmarks_dict))]
    return landmarks_list

def convert_landmarks_to_numpy(landmarks_dict):
    """
    将关键点字典转换为numpy数组
    
    Args:
        landmarks_dict (dict): 关键点字典
    
    Returns:
        numpy.ndarray: 关键点坐标数组
    """
    # 方法1：使用列表推导
    landmarks_array = np.array([list(landmarks_dict[i]) for i in range(len(landmarks_dict))])
    
    # 方法2：使用numpy从列表创建
    # landmarks_array = np.fromiter(
    #     (coord for key in sorted(landmarks_dict.keys()) 
    #      for coord in landmarks_dict[key]), 
    #     dtype=float
    # ).reshape(-1, 3)
    
    return landmarks_array