import csv


def get_data_types(data):
    data_types = []
    for value in data:
        data_types.append(type(value).__name__)
    unique_data_types = set(data_types)
    return unique_data_types


with open('train.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    lines = list(reader)

header = lines[0]
data = lines[1:]

# 创建一个字典来存储每一列的数据类型
column_types = {}

# 初始化字典的值为一个空列表
for column in header:
    column_types[column] = []

# 遍历每一列并获取数据类型
for column_index, column_name in enumerate(header):
    column_data = [row[column_index] for row in data]
    column_types[column_name] = get_data_types(column_data)

# 打印每一列的具体数据类型
for column, types in column_types.items():
    print(f"Column '{column}' has data types: {types}")
