import csv

# 打开 CSV 文件并读取其内容
with open('val_val.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    lines = list(reader)

# 打开 CSV 文件并读取内容
with open('en_test.csv', 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)

    count = 0  # 用于计数已读取的行数

    # 遍历每一行并获取第三列和第四列的数据
    for row in reader:
        if count >= 50000:  # 当读取到50000条时停止读取
            break

        answer = row[2]  # 第三列的索引为2，因为索引从0开始
        l = row[3]  # 将第四列的数据转换为整数类型
        rows = [answer, l]  # 不需要再使用 f-string 格式化

        # 处理第三列和第四列的数据
        lines.insert(1, rows)
        print(rows)

        count += 1  # 每读取一行，计数器加1

# 将修改后的内容写回到 train.csv 文件中
with open('val_val.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(lines)
