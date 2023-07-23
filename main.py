import pickle

# 指定.pkl文件路径
file_path = "AoM数据集资源/VG_train.pkl"

# 打开.pkl文件并加载数据
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 打印数据内容
print(data)
