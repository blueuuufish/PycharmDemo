import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('text_test.csv')

# 仅保留前五行数据
df = df.head(2)

# 如果你想保存修改后的数据到新的 CSV 文件，可以使用以下代码：
df.to_csv('test_test.csv', index=False)
