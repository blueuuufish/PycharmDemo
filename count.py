import pandas as pd

# 读取train.csv文件
df = pd.read_csv('train.csv')

# 定义初始最大字数为0
max_length = 0

# 遍历text列，计算最大字数
for text in df['text']:
    length = len(text)
    if length > max_length:
        max_length = length

# 输出最大字数
print("最大字数:", max_length)