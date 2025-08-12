# import numpy as np
# import torch  # Import PyTorch
# from pykeen.pipeline import pipeline
# from pykeen.triples import TriplesFactory

# # 训练集 - 示例数据, 注意替换为你的实际数据
# train_data = np.array([
#     ['1', '交易', '5'],
#     ['2', '交易', '5'],
#     ['3', '交易', '6'],
#     ['4', '交易', '8'],
#     ['5', '交易', '21']
# ])

# # 测试集 - 示例数据, 注意替换为你的实际数据
# test_data = np.array([
#     ['1', '交易', '6'],
#     ['1', '交易', '5'],
#     ['1', '交易', '8'],
#     ['2', '交易', '21'],
#     ['3', '交易', '5'],
# ])

# # 创建 TriplesFactory
# tf_train = TriplesFactory.from_labeled_triples(train_data)
# tf_test = TriplesFactory.from_labeled_triples(test_data)

# # 设置 pipeline
# result = pipeline(
#     model='ComplEx',  # 使用 ComplEx 模型
#     training=tf_train,
#     testing=tf_test,
#     training_kwargs={
#         'batch_size': 2,  # 设置一个合理的批次大小
#         'drop_last': False  # 确保不丢失数据
#     },
# )

# # 查看测试结果
# for head, relation, tail in test_data:
#     # Convert strings to integer indices using the entity to id mapping from the training triples factory
#     head_id = tf_train.entity_to_id[head]
#     relation_id = tf_train.relation_to_id[relation]
#     tail_id = tf_train.entity_to_id[tail]

#     # Convert the NumPy array to a PyTorch tensor
#     hrt_batch = torch.tensor([[head_id, relation_id, tail_id]])

#     # Get the score for the specific triple
#     score = result.model.score_hrt(hrt_batch=hrt_batch)
#     print(f"Score for {head} - {relation} - {tail}: {score.item()}")

#模型测试
# import numpy as np
# import pandas as pd
# import torch
# from pykeen.pipeline import pipeline
# from pykeen.triples import TriplesFactory

# # 假设训练集和测试集文件路径
# train_data_path = '河流拓扑结构.xlsx'
# test_data_path = 'test1.xlsx'

# # 从Excel文件读取训练集和测试集数据
# # 提取训练数据中的所需列
# train_data = pd.read_excel(train_data_path, header=None, usecols=[8, 9, 11], dtype={8: str, 9: str, 11: str}).values
# test_data = pd.read_excel(test_data_path, header=None, dtype={0: str, 1: str, 2: str}).values

# # 打印一部分数据以确认列数
# print("Training data:")
# print(train_data[:5])
# print("\nTest data:")
# print(test_data[:5])

# # 确保训练数据和测试数据的列数相同
# assert train_data.shape[1] == test_data.shape[1], "训练数据和测试数据的列数不匹配"

# # 将训练集和测试集合并
# all_data = np.concatenate((train_data, test_data), axis=0)

# # 创建 TriplesFactory
# tf_all = TriplesFactory.from_labeled_triples(all_data)

# # 使用所有数据创建训练集和测试集
# tf_train = TriplesFactory.from_labeled_triples(train_data, entity_to_id=tf_all.entity_to_id, relation_to_id=tf_all.relation_to_id)
# tf_test = TriplesFactory.from_labeled_triples(test_data, entity_to_id=tf_all.entity_to_id, relation_to_id=tf_all.relation_to_id)

# # 设置 pipeline
# result = pipeline(
#     model='ComplEx',
#     training=tf_train,
#     testing=tf_test,
#     training_kwargs={
#         'batch_size': 1,  # 减小批量大小
#         'drop_last': False  # 确保不丢失数据
#     },
# )

# # 查看测试结果
# for head, relation, tail in test_data:
#     # Convert strings to integer indices using the entity to id mapping from the training triples factory
#     head_id = tf_train.entity_to_id[head]
#     relation_id = tf_train.relation_to_id[relation]
#     tail_id = tf_train.entity_to_id[tail]

#     # Convert the NumPy array to a PyTorch tensor
#     hrt_batch = torch.tensor([[head_id, relation_id, tail_id]])

#     # Get the score for the specific triple
#     score = result.model.score_hrt(hrt_batch=hrt_batch)
#     print(f"Score for {head} - {relation} - {tail}: {score.item()}")

# import numpy as np
# import pandas as pd
# import torch
# from pykeen.pipeline import pipeline
# from pykeen.triples import TriplesFactory

# # 假设训练集和测试集文件路径
# train_data_path = '河流拓扑结构.xlsx'
# test_data_path = 'test1.xlsx'

# # 加载训练数据，跳过第一行
# train_data = pd.read_excel(train_data_path, header=None, usecols=[8, 9, 11], skiprows=1)
# train_data.columns = ['head', 'tail', 'relation']

# # 加载测试数据，跳过第一行
# test_data = pd.read_excel(test_data_path, header=None, usecols=[0, 1, 2], skiprows=1)
# test_data.columns = ['head', 'tail', 'relation']

# # 确保数据类型一致，将所有列转换为字符串类型
# train_data = train_data.astype(str)
# test_data = test_data.astype(str)

# # 将关系列移动到第二列以符合 (head, relation, tail) 的顺序
# train_data = train_data[['head', 'relation', 'tail']]
# test_data = test_data[['head', 'relation', 'tail']]

# # 合并训练和测试数据
# all_data = pd.concat([train_data, test_data], ignore_index=True)

# # 输出清理后的数据
# print(all_data.head())

# # 将数据转换为二维数组
# all_data = all_data.values

# # 创建 TriplesFactory
# tf_all = TriplesFactory.from_labeled_triples(all_data)

# # 使用所有数据创建训练集和测试集
# tf_train = TriplesFactory.from_labeled_triples(train_data.values, entity_to_id=tf_all.entity_to_id, relation_to_id=tf_all.relation_to_id)
# tf_test = TriplesFactory.from_labeled_triples(test_data.values, entity_to_id=tf_all.entity_to_id, relation_to_id=tf_all.relation_to_id)

# # 设置 pipeline
# result = pipeline(
#     model='ComplEx',
#     training=tf_train,
#     testing=tf_test,
#     training_kwargs={
#         'batch_size': 1,  # 减小批量大小
#         'drop_last': False  # 确保不丢失数据
#     },
# )

# # 获取所有实体和关系
# entities = list(tf_train.entity_to_id.keys())
# relations = list(tf_train.relation_to_id.keys())

# # 打印所有实体
# print("Entities:")
# for entity in entities:
#     print(entity)

# # 打印所有关系.quit
# print("\nRelations:")
# for relation in relations:
#     print(relation)

# # 用于保存预测结果的列表
# predictions = []

# # 遍历所有实体
# for head in entities:
#     head_id = tf_train.entity_to_id[head]
    
#     # 创建一个空的Tensor来存储预测分数
#     scores = torch.zeros(len(entities))
    
#     # 对于每个目标实体，计算预测分数
#     for i, tail in enumerate(entities):
#         if head != tail:
#             tail_id = tf_train.entity_to_id[tail]
            
#             # 选择一个关系来测试，可以修改 relation_id 以测试不同的关系
#             relation_id = 0  # 例如，你可以选择某个具体关系的ID
#             hrt_batch = torch.tensor([[head_id, relation_id, tail_id]])
#             score = result.model.score_hrt(hrt_batch=hrt_batch)
#             scores[i] = score.item()

#     # 获取预测分数最高的前三个尾实体
#     top_tails_indices = torch.topk(scores, k=3).indices  # 选择前3个最高分的实体ID
    
#     # 只保留得分最高的前三个关系
#     top_tails = [entities[i] for i in top_tails_indices]
#     top_scores = scores[top_tails_indices]
    
#     # 添加到预测结果
#     for tail, score in zip(top_tails, top_scores):
#         predictions.append([head, tail, score.item()])

# # 将预测结果转换为DataFrame并保存到CSV文件
# df_predictions = pd.DataFrame(predictions, columns=['Head', 'Tail', 'Score'])

# # 打印预测结果
# print(df_predictions)

# # 将预测结果保存到CSV文件
# df_predictions.to_csv('top3_scores.csv', index=False)

# print("Top 3 scores have been saved to 'top3_scores.csv'.")


#rgcn 百分之88
# import pandas as pd
# import torch
# from pykeen.pipeline import pipeline
# from pykeen.triples import TriplesFactory

# # 假设训练集和测试集文件路径
# train_data_path = 'train_tradedata.csv'
# test_data_path = 'test_tradedata.csv'

# # 加载训练数据
# train_data = pd.read_csv(train_data_path)
# train_data = train_data.drop(columns=['Sold_wec'])  # 去除 'Sold_wec' 列
# train_data.columns = ['head', 'tail', 'relation']

# # 加载测试数据，并去除 'Sold_wec' 列
# test_data = pd.read_csv(test_data_path)
# test_data = test_data.drop(columns=['Sold_wec'])  # 去除 'Sold_wec' 列
# test_data.columns = ['head', 'tail', 'relation']

# # 确保数据类型一致，将所有列转换为字符串类型
# train_data = train_data.astype(str)
# test_data = test_data.astype(str)

# # 将关系列移动到第二列以符合 (head, relation, tail) 的顺序
# train_data = train_data[['head', 'relation', 'tail']]
# test_data = test_data[['head', 'relation', 'tail']]

# # 创建 TriplesFactory
# tf_train = TriplesFactory.from_labeled_triples(train_data.values)
# tf_test = TriplesFactory.from_labeled_triples(test_data.values, 
#                                               entity_to_id=tf_train.entity_to_id, 
#                                               relation_to_id=tf_train.relation_to_id)

# # 使用 CompGCN 模型进行训练和推理
# result = pipeline(
#     model='RGCN',
#     training=tf_train,
#     testing=tf_test,
#     training_kwargs={
#         'num_epochs': 260,  # 训练的轮数
#         'batch_size': 32,  # 批处理大小
#     },
# )

# # 获取所有实体和关系
# entities = list(tf_train.entity_to_id.keys())

# # 用于保存每个head节点得分最高的三个预测结果的列表
# top_predictions = []

# # 遍历所有可能的 (head, tail) 对
# for head in entities:
#     head_id = tf_train.entity_to_id[head]
#     scores = []
    
#     for tail in entities:
#         if head != tail:  # 排除自环
#             tail_id = tf_train.entity_to_id[tail]
#             relation_id = 0  # 假设只有一个关系类型
            
#             # 创建 HRT batch
#             hrt_batch = torch.tensor([[head_id, relation_id, tail_id]])
            
#             # 获取模型对该三元组的评分
#             score = result.model.score_hrt(hrt_batch=hrt_batch)
            
#             # 保存当前 (head, tail) 及其得分
#             scores.append([head, tail, score.item()])
    
#     # 对当前 head 的所有 (tail, score) 对进行排序，取前三
#     scores.sort(key=lambda x: x[2], reverse=True)  # 修正了排序方法
#     top_predictions.extend(scores[:3])

# # 将预测结果转换为DataFrame
# df_top_predictions = pd.DataFrame(top_predictions, columns=['Head', 'Tail', 'Score'])

# # 打印每个节点得分最高的三个预测结果
# print("Top 3 Predictions for Each Head Node:")
# print(df_top_predictions)

# # 将这些结果保存到CSV文件
# df_top_predictions.to_csv('top_3_predictions_per_head.csv', index=False)
# print("Top 3 predictions for each head node have been saved to 'top_3_predictions_per_head.csv'.")

# # 对比测试集和预测结果
# test_data_set = set(map(tuple, test_data[['head', 'tail']].values))
# predicted_data_set = set(map(tuple, df_top_predictions[['Head', 'Tail']].values))

# # 找到测试集中的真实对比结果
# correct_predictions = test_data_set & predicted_data_set

# print(f"\n共有 {len(correct_predictions)} 个预测结果在测试集中找到匹配:")
# for item in correct_predictions:
#     print(item)






import numpy as np
import pandas as pd
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# 假设训练集和测试集文件路径
train_data_path = 'train_tradedata.csv'
test_data_path = 'test_tradedata.csv'

# 加载训练数据
train_data = pd.read_csv(train_data_path)
train_data = train_data.drop(columns=['Sold_wec'])  # 去除 'Sold_wec' 列
train_data.columns = ['head', 'tail', 'relation']

# 加载测试数据，并去除 'Sold_wec' 列
test_data = pd.read_csv(test_data_path)
test_data = test_data.drop(columns=['Sold_wec'])  # 去除 'Sold_wec' 列
test_data.columns = ['head', 'tail', 'relation']

# 确保数据类型一致，将所有列转换为字符串类型
train_data = train_data.astype(str)
test_data = test_data.astype(str)

# 将关系列移动到第二列以符合 (head, relation, tail) 的顺序
train_data = train_data[['head', 'relation', 'tail']]
test_data = test_data[['head', 'relation', 'tail']]

# 合并训练和测试数据
all_data = pd.concat([train_data, test_data], ignore_index=True)

# 输出清理后的数据
print("All Data:\n", all_data.head())

# 将数据转换为二维数组
all_data = all_data.values

# 创建 TriplesFactory
tf_all = TriplesFactory.from_labeled_triples(all_data)

# 使用所有数据创建训练集和测试集
tf_train = TriplesFactory.from_labeled_triples(train_data.values, entity_to_id=tf_all.entity_to_id, relation_to_id=tf_all.relation_to_id)
tf_test = TriplesFactory.from_labeled_triples(test_data.values, entity_to_id=tf_all.entity_to_id, relation_to_id=tf_all.relation_to_id)

# 设置 pipeline
result = pipeline(
    model='ComplEx',
    training=tf_train,
    testing=tf_test,
    training_kwargs={
        'batch_size': 1,  # 减小批量大小
        'drop_last': False,  # 确保不丢失数据
        'num_epochs': 200,  # 初始设置为50轮
    },
)

# 获取所有实体和关系
entities = list(tf_train.entity_to_id.keys())
relations = list(tf_train.relation_to_id.keys())

# 打印所有实体
print("Entities:")
for entity in entities:
    print(entity)

# 打印所有关系
print("\nRelations:")
for relation in relations:
    print(relation)

# 用于保存预测结果的列表
predictions = []

# 遍历所有实体
for head in entities:
    head_id = tf_train.entity_to_id[head]
    
    # 创建一个空的Tensor来存储预测分数
    scores = torch.zeros(len(entities))
    
    # 对于每个目标实体，计算预测分数
    for i, tail in enumerate(entities):
        if head != tail:
            tail_id = tf_train.entity_to_id[tail]
            
            # 选择一个关系来测试，可以修改 relation_id 以测试不同的关系
            relation_id = 0  # 例如，你可以选择某个具体关系的ID
            hrt_batch = torch.tensor([[head_id, relation_id, tail_id]])
            score = result.model.score_hrt(hrt_batch=hrt_batch)
            scores[i] = score.item()

    # 获取预测分数最高的前三个尾实体
    top_tails_indices = torch.topk(scores, k=3).indices  # 选择前3个最高分的实体ID
    
    # 只保留得分最高的前三个关系
    top_tails = [entities[i] for i in top_tails_indices]
    top_scores = scores[top_tails_indices]
    
    # 添加到预测结果
    for tail, score in zip(top_tails, top_scores):
        predictions.append([head, tail, score.item()])

# 将预测结果转换为DataFrame并保存到CSV文件
df_predictions = pd.DataFrame(predictions, columns=['Head', 'Tail', 'Score'])

# 打印预测结果
print("Predictions:")
print(df_predictions)

# 将预测结果保存到CSV文件
df_predictions.to_csv('top3_scores.csv', index=False)
print("Top 3 scores have been saved to 'top3_scores.csv'.")

# 对比测试集和预测结果
test_data_set = set(map(tuple, test_data[['head', 'tail']].values))
predicted_data_set = set(map(tuple, df_predictions[['Head', 'Tail']].values))

# 找到测试集中的真实对比结果
correct_predictions = test_data_set & predicted_data_set

print(f"\n共有 {len(correct_predictions)} 个预测结果在测试集中找到匹配:")
for item in correct_predictions:
    print(item)


# import pandas as pd
# import torch
# from pykeen.pipeline import pipeline
# from pykeen.triples import TriplesFactory

# # 假设训练集和测试集文件路径
# train_data_path = 'train_tradedata.csv'
# test_data_path = 'test_tradedata.csv'

# # 加载训练数据
# train_data = pd.read_csv(train_data_path)
# train_data = train_data.drop(columns=['Sold_wec'])  # 去除 'Sold_wec' 列
# train_data.columns = ['head', 'tail', 'relation']

# # 加载测试数据，并去除 'Sold_wec' 列
# test_data = pd.read_csv(test_data_path)
# test_data = test_data.drop(columns=['Sold_wec'])  # 去除 'Sold_wec' 列
# test_data.columns = ['head', 'tail', 'relation']

# # 确保数据类型一致，将所有列转换为字符串类型
# train_data = train_data.astype(str)
# test_data = test_data.astype(str)

# # 将关系列移动到第二列以符合 (head, relation, tail) 的顺序
# train_data = train_data[['head', 'relation', 'tail']]
# test_data = test_data[['head', 'relation', 'tail']]

# # 创建 TriplesFactory，并启用反向三元组的创建
# tf_train = TriplesFactory.from_labeled_triples(
#     train_data.values, 
#     create_inverse_triples=True
# )
# tf_test = TriplesFactory.from_labeled_triples(
#     test_data.values, 
#     entity_to_id=tf_train.entity_to_id, 
#     relation_to_id=tf_train.relation_to_id, 
#     create_inverse_triples=True
# )

# # 使用 CompGCN 模型进行训练和推理
# result = pipeline(
#     model='CompGCN',
#     training=tf_train,
#     testing=tf_test,
#     training_kwargs={
#         'num_epochs': 280,  # 训练的轮数
#         'batch_size': 32,  # 批处理大小
#     },
# )

# # 获取所有实体和关系
# entities = list(tf_train.entity_to_id.keys())

# # 用于保存每个head节点得分最高的三个预测结果的列表
# top_predictions = []

# # 遍历所有可能的 (head, tail) 对
# for head in entities:
#     head_id = tf_train.entity_to_id[head]
#     scores = []
    
#     for tail in entities:
#         if head != tail:  # 排除自环
#             tail_id = tf_train.entity_to_id[tail]
#             relation_id = 0  # 假设只有一个关系类型
            
#             # 创建 HRT batch
#             hrt_batch = torch.tensor([[head_id, relation_id, tail_id]])
            
#             # 获取模型对该三元组的评分
#             score = result.model.score_hrt(hrt_batch=hrt_batch)
            
#             # 保存当前 (head, tail) 及其得分
#             scores.append([head, tail, score.item()])
    
#     # 对当前 head 的所有 (tail, score) 对进行排序，取前三
#     scores.sort(key=lambda x: x[2], reverse=True)
#     top_predictions.extend(scores[:3])

# # 将预测结果转换为DataFrame
# df_top_predictions = pd.DataFrame(top_predictions, columns=['Head', 'Tail', 'Score'])

# # 打印每个节点得分最高的三个预测结果
# print("Top 3 Predictions for Each Head Node:")
# print(df_top_predictions)

# # 将这些结果保存到CSV文件
# df_top_predictions.to_csv('top_3_predictions_per_head.csv', index=False)
# print("Top 3 predictions for each head node have been saved to 'top_3_predictions_per_head.csv'.")

# # 对比测试集和预测结果
# test_data_set = set(map(tuple, test_data[['head', 'tail']].values))
# predicted_data_set = set(map(tuple, df_top_predictions[['Head', 'Tail']].values))

# # 找到测试集中的真实对比结果
# correct_predictions = test_data_set & predicted_data_set

# print(f"\n共有 {len(correct_predictions)} 个预测结果在测试集中找到匹配:")
# for item in correct_predictions:
#     print(item)
