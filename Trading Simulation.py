


# import pandas as pd
# import numpy as np

# # 假设有160个河段
# np.random.seed(42)
# num_segments = 160

# # 生成[-30, -1]和[1, 30]之间的随机整数，不包含0
# wec_values = np.random.choice(
#     np.concatenate((np.arange(-30, 0), np.arange(1, 31))), 
#     size=num_segments, 
#     replace=True
# )

# # 创建DataFrame存储河段和对应的WEC值
# original_df = pd.DataFrame({'segment': range(1, num_segments+1), 'wec_value': wec_values})

# # 定义交易函数
# def trade_wec(df):
#     # 找出所有需要购买WEC值的河段（即WEC值大于0）
#     buyers = df[df['wec_value'] > 0].copy()
#     # 找出所有可以出售WEC值的河段（即WEC值小于0）
#     sellers = df[df['wec_value'] < 0].copy()
    
#     # 初始化交易记录
#     trades = []
    
#     # 遍历买家
#     for index, buyer in buyers.iterrows():
#         # 计算买家需要购买的WEC值
#         need_to_buy = abs(buyer['wec_value'])
#         num_purchases = 0  # 初始化买家购买的次数
        
#         # 遍历卖家，确保最多购买3次
#         for seller_index, seller in sellers.iterrows():
#             if num_purchases >= 3:
#                 break  # 一旦达到购买次数上限，跳出循环
            
#             # 如果卖家有足够的WEC值出售，且交易值至少为1
#             if abs(seller['wec_value']) >= need_to_buy >= 1:
#                 # 进行交易
#                 seller['wec_value'] += need_to_buy
#                 buyer['wec_value'] = 0
                
#                 # 记录交易
#                 trades.append((buyer['segment'], seller['segment'], need_to_buy, "交易"))
                
#                 # 更新DataFrame
#                 df.loc[index, 'wec_value'] = buyer['wec_value']
#                 df.loc[seller_index, 'wec_value'] = seller['wec_value']
                
#                 num_purchases += 1
#                 break  # 一旦交易完成，跳出循环
            
#             # 如果卖家的WEC值不足以完全满足买家的需求，但至少为1
#             elif abs(seller['wec_value']) > 0 and need_to_buy >= 1:
#                 trade_value = abs(seller['wec_value'])
#                 # 进行部分交易
#                 buyer['wec_value'] -= trade_value
#                 seller['wec_value'] = 0
                
#                 # 记录交易
#                 if trade_value >= 1:
#                     trades.append((buyer['segment'], seller['segment'], trade_value, "交易"))
                
#                 # 更新DataFrame
#                 df.loc[index, 'wec_value'] = buyer['wec_value']
#                 df.loc[seller_index, 'wec_value'] = seller['wec_value']
                
#                 num_purchases += 1
#                 # 继续寻找下一个卖家
#                 continue
    
#     # 确保交易后的WEC值为非负数
#     for index, row in df.iterrows():
#         if row['wec_value'] < 0:
#             print(f"Error: 河段{row['segment']}的WEC值为负数 {row['wec_value']}")
    
#     return trades

# # 模拟九轮交易并保存到训练集
# all_train_trades = []

# # 每次循环都从原始数据创建副本，以确保每一轮交易独立进行
# for _ in range(9):
#     df = original_df.copy()  # 使用副本来进行交易
#     trades = trade_wec(df)
#     all_train_trades.extend(trades)

# # 将九轮交易记录转换成DataFrame并保存到train_tradedata.csv
# train_trades_df = pd.DataFrame(all_train_trades, columns=['Buyer', 'Seller', 'Sold_wec', 'Relation'])
# train_trades_df.to_csv('train_tradedata.csv', index=False)

# # 进行一轮新的交易并保存到测试集
# df = original_df.copy()  # 重新创建数据副本
# test_trades = trade_wec(df)
# test_trades_df = pd.DataFrame(test_trades, columns=['Buyer', 'Seller', 'Sold_wec', 'Relation'])
# test_trades_df.to_csv('test_tradedata.csv', index=False)

# print("训练集交易记录已保存到 'train_tradedata.csv' 文件。")
# print("测试集交易记录已保存到 'test_tradedata.csv' 文件。")


# import pandas as pd
# import numpy as np

# # 假设有160个河段
# np.random.seed(42)
# num_segments = 160

# # 生成[-30, -1]和[1, 30]之间的随机整数，不包含0
# wec_values = np.random.choice(
#     np.concatenate((np.arange(-30, 0), np.arange(1, 31))), 
#     size=num_segments, 
#     replace=True
# )

# # 创建DataFrame存储河段和对应的WEC值
# original_df = pd.DataFrame({'segment': range(1, num_segments+1), 'wec_value': wec_values})

# # 定义一个简单的距离函数（假设河段之间的距离是它们的序号差异）
# def segment_distance(buyer_segment, seller_segment):
#     return abs(buyer_segment - seller_segment)

# # 定义交易函数
# def trade_wec(df):
#     buyers = df[df['wec_value'] > 0].copy()
#     sellers = df[df['wec_value'] < 0].copy()
    
#     trades = []
    
#     for index, buyer in buyers.iterrows():
#         need_to_buy = abs(buyer['wec_value'])
#         num_purchases = 0
        
#         # 计算与每个卖家的距离
#         sellers['distance'] = sellers['segment'].apply(lambda x: segment_distance(buyer['segment'], x))
        
#         # 根据距离从近到远排序卖家
#         sellers = sellers.sort_values(by='distance')
        
#         for seller_index, seller in sellers.iterrows():
#             if num_purchases >= 3:
#                 break
            
#             if abs(seller['wec_value']) >= need_to_buy >= 1:
#                 seller['wec_value'] += need_to_buy
#                 buyer['wec_value'] = 0
#                 trades.append((buyer['segment'], seller['segment'], need_to_buy, "交易"))
#                 df.loc[index, 'wec_value'] = buyer['wec_value']
#                 df.loc[seller_index, 'wec_value'] = seller['wec_value']
#                 num_purchases += 1
#                 break
            
#             elif abs(seller['wec_value']) > 0 and need_to_buy >= 1:
#                 trade_value = abs(seller['wec_value'])
#                 buyer['wec_value'] -= trade_value
#                 seller['wec_value'] = 0
#                 if trade_value >= 1:
#                     trades.append((buyer['segment'], seller['segment'], trade_value, "交易"))
#                 df.loc[index, 'wec_value'] = buyer['wec_value']
#                 df.loc[seller_index, 'wec_value'] = seller['wec_value']
#                 num_purchases += 1
#                 continue
    
#     return trades

# # 模拟九轮交易并保存到训练集
# all_train_trades = []
# for _ in range(20):
#     df = original_df.copy()
#     trades = trade_wec(df)
#     all_train_trades.extend(trades)

# train_trades_df = pd.DataFrame(all_train_trades, columns=['Buyer', 'Seller', 'Sold_wec', 'Relation'])
# train_trades_df.to_csv('train_tradedata.csv', index=False)

# # 进行一轮新的交易并保存到测试集
# df = original_df.copy()
# test_trades = trade_wec(df)
# test_trades_df = pd.DataFrame(test_trades, columns=['Buyer', 'Seller', 'Sold_wec', 'Relation'])
# test_trades_df.to_csv('test_tradedata.csv', index=False)

# print("训练集交易记录已保存到 'train_tradedata.csv' 文件。")
# print("测试集交易记录已保存到 'test_tradedata.csv' 文件。")


import pandas as pd
import numpy as np
import networkx as nx

# 读取河流拓扑结构数据
topology_df = pd.read_excel('河流拓扑结构.xlsx')

# 构建图结构
G = nx.Graph()
for _, row in topology_df.iterrows():
    G.add_edge(row['FROM_NODE'], row['TO_NODE'])

# 假设有160个河段
np.random.seed(42)
num_segments = 160

# 定义基于图的最短路径距离计算函数
def segment_distance(buyer_segment, seller_segment):
    try:
        return nx.shortest_path_length(G, source=buyer_segment, target=seller_segment)
    except nx.NetworkXNoPath:
        return np.inf

# 定义交易函数
def trade_wec(df):
    buyers = df[df['wec_value'] > 0].copy()
    sellers = df[df['wec_value'] < 0].copy()

    trades = []

    for index, buyer in buyers.iterrows():
        need_to_buy = abs(buyer['wec_value'])
        num_purchases = 0

        sellers['distance'] = sellers['segment'].apply(lambda x: segment_distance(buyer['segment'], x))

        sellers = sellers.sort_values(by='distance')

        candidate_sellers = sellers.head(np.random.randint(2, 6))

        for seller_index, seller in candidate_sellers.iterrows():
            if num_purchases >= 3:
                break

            if abs(seller['wec_value']) >= need_to_buy >= 1:
                seller['wec_value'] += need_to_buy
                buyer['wec_value'] = 0
                trades.append((buyer['segment'], seller['segment'], need_to_buy, "交易"))
                df.loc[index, 'wec_value'] = buyer['wec_value']
                df.loc[seller_index, 'wec_value'] = seller['wec_value']
                num_purchases += 1
                break

            elif abs(seller['wec_value']) > 0 and need_to_buy >= 1:
                trade_value = abs(seller['wec_value'])
                buyer['wec_value'] -= trade_value
                seller['wec_value'] = 0
                if trade_value >= 1:
                    trades.append((buyer['segment'], seller['segment'], trade_value, "交易"))
                df.loc[index, 'wec_value'] = buyer['wec_value']
                df.loc[seller_index, 'wec_value'] = seller['wec_value']
                num_purchases += 1
                continue

    return trades

# 模拟九轮交易并保存到训练集
all_train_trades = []

for _ in range(20):
    # 每一轮都重新生成 WEC 值
    wec_values = np.random.choice(
        np.concatenate((np.arange(-30, 0), np.arange(1, 31))),
        size=num_segments,
        replace=True
    )
    df = pd.DataFrame({'segment': range(1, num_segments + 1), 'wec_value': wec_values})
    
    trades = trade_wec(df)
    all_train_trades.extend(trades)

train_trades_df = pd.DataFrame(all_train_trades, columns=['Buyer', 'Seller', 'Sold_wec', 'Relation'])
train_trades_df.to_csv('train_tradedata.csv', index=False)

# 进行一轮新的交易并保存到测试集
wec_values = np.random.choice(
    np.concatenate((np.arange(-30, 0), np.arange(1, 31))),
    size=num_segments,
    replace=True
)
df = pd.DataFrame({'segment': range(1, num_segments + 1), 'wec_value': wec_values})

test_trades = trade_wec(df)
test_trades_df = pd.DataFrame(test_trades, columns=['Buyer', 'Seller', 'Sold_wec', 'Relation'])
test_trades_df.to_csv('test_tradedata.csv', index=False)

print("训练集交易记录已保存到 'train_tradedata.csv' 文件。")
print("测试集交易记录已保存到 'test_tradedata.csv' 文件。")
