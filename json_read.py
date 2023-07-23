import json

# 打开并读取JSON文件
with open('train.json', 'r') as f:
    data = json.load(f)

# 对每个数据项进行处理
for item in data:
    # 提取并打印'words'字段（如果存在）
    if 'words' in item:
        sentence = ' '.join(item['words'])
        print(sentence, end=' ')
        print()

    # 提取并打印'aspects'字段中的'polarity'和'term'（如果存在）
    if 'aspects' in item:
        for aspect in item['aspects']:
            print(f"情感判断: {aspect['polarity']}, 词语: {' '.join(aspect['term'])}")

    # 打印一个空行以分隔不同的数据项
    print()
