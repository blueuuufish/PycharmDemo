import pandas as pd

# 1. 读取 CSV 文件到 DataFrame
df = pd.read_csv('test.csv')

# 2. 创建一个新的 `id` 列
df['id'] = range(1, len(df) + 1)

# 3. 创建 `question` 和 `answer` 列
df['question'] = df['text']
df['answer'] = df['text']

# 4. 重新排列列的顺序
df = df[['id', 'question', 'answer', 'label']]

# 5. 保存修改后的 DataFrame 到新的 CSV 文件
df.to_csv('test_test.csv', index=False)
