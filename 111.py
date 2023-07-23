import json

# 打开并读取JSON文件
with open('train.json', 'r') as f:
    data = json.load(f)

# 提取每个数据中的'words'字段，并将其转换为句子
sentences = [' '.join(item['words']) for item in data if 'words' in item]

# 打印提取的句子
for sentence in sentences:
    print(sentence)

# 提取并打印每个数据项的'aspect'和'label'
for item in data:
    if 'aspect' in item and 'label' in item:
        print(f"Aspect: {item['aspect']}, Label: {item['label']}")