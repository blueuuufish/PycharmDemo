import torch
from layers import GraphLearner

# 初始化一个Tensor
node_features = torch.randn(16, 94, 768)

# 创建一个GraphLearner实例
graph_learner = GraphLearner(input_size=768, hidden_size=128, graph_type='KNN', top_k=10, num_pers=4, metric_type="attention")

# 使用GraphLearner处理Tensor
new_node_features, learned_adj = graph_learner(node_features)

# 打印结果
print(new_node_features)
print(learned_adj)
print("--------------------------------------")
print("new_node_features shape:", new_node_features.shape)
print("learned_adj shape:", learned_adj.shape)
print("--------------------------------------")
print("new_node_features shape:", new_node_features.shape, "dtype:", new_node_features.dtype, "device:", new_node_features.device)
print("learned_adj shape:", learned_adj.shape, "dtype:", learned_adj.dtype, "device:", learned_adj.device)

