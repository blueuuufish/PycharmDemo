import json

# 读取JSON文件
with open('train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理每个数据项
for item in data:
    words = item['words']
    aspects = item['aspects']

    # 从后向前遍历aspects
    for aspect in reversed(aspects):
        from_index = aspect['from']
        to_index = aspect['to']
        merged_words = ' '.join(words[from_index:to_index])
        words = words[:from_index] + [merged_words] + words[to_index:]

    # 更新数据项的words
    item['words'] = words

    # 更新term并根据新的words更新from和to
    for aspect in aspects:
        aspect['term'] = [' '.join(aspect['term'])]
        if aspect['term'][0] in words:
            new_index = words.index(aspect['term'][0])
            aspect['from'] = new_index
            aspect['to'] = new_index + 1

# 写入更新后的数据到新的JSON文件
with open('updated_train.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)
