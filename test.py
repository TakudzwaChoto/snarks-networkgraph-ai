# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier

# # Load data using pandas
# df = pd.read_csv('PimaIndiansdiabetes.csv')

# # Split data into features (X) and target variable (Y)
# X = df.iloc[:, :-1].values
# Y = df.iloc[:, -1].values

# # Split data into train and test sets
# seed = 7
# test_size = 0.33
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# # Fit model on training data
# model = XGBClassifier()
# model.fit(X_train, y_train)

# # Make predictions for test data
# predictions = model.predict(X_test)

# # Evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print(f'Accuracy: {accuracy * 100:.2f}%')


#CatBoost 
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from catboost import CatBoostClassifier

# # Load data using pandas
# df = pd.read_csv('PimaIndiansdiabetes.csv')

# # Split data into features (X) and target variable (Y)
# X = df.iloc[:, :-1].values
# Y = df.iloc[:, -1].values

# # Split data into train and test sets
# seed = 7
# test_size = 0.2
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# # Fit model on training data
# # Initialize the CatBoostClassifier with your preferred parameters
# model = CatBoostClassifier(verbose=False) # verbose=False to suppress output during training
# # You can set more parameters here like iterations, learning_rate, depth, etc.

# model.fit(X_train, y_train)

# # Make predictions for test data
# predictions = model.predict(X_test)

# # Evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print(f'Accuracy: {accuracy * 100:.2f}%')

#预测1：回归任务
# import pandas as pd
# from catboost import CatBoostRegressor, Pool
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # data = pd.read_excel('河道氨氮统计数据--环境容量.xlsx', sheet_name='Sheet1')
# data = pd.read_csv('housing.csv')
# # 假设目标变量是我们想要预测的氨氮浓度
# y = data['MEDV']
# # 特征选择，这里我们选择除氨氮浓度外的数值型特征
# X = data.drop(['MEDV'], axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = CatBoostRegressor(iterations=1000, learning_rate=0.1, loss_function='RMSE', verbose=True)
# # 使用Pool包装数据，以适应CatBoost的输入格式
# train_pool = Pool(data=X_train, label=y_train)
# test_pool = Pool(data=X_test, label=y_test)
# model.fit(train_pool)

# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# for i in range(len(y_test)):
#     print(f"Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}")

#聚类kmeans
# import pandas as pd
# from catboost import CatBoostRegressor, Pool
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.cluster import KMeans

# # 读取数据
# data = pd.read_excel('河道氨氮统计数据--环境容量.xlsx', sheet_name='Sheet1')

# # 假设目标变量是我们想要预测的氨氮浓度
# y = data['氨氮浓度']

# # 特征选择，这里我们选择除氨氮浓度外的数值型特征
# X = data.drop(['氨氮浓度', 'RCH'], axis=1)

# # 应用KMeans聚类
# n_clusters = 5  # 你可以根据数据情况选择不同的簇数量
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# X['cluster'] = kmeans.fit_predict(X)

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 使用Pool包装数据，以适应CatBoost的输入格式
# train_pool = Pool(data=X_train, label=y_train)
# test_pool = Pool(data=X_test, label=y_test)

# # 初始化并训练模型
# model = CatBoostRegressor(iterations=1000, learning_rate=0.1, loss_function='RMSE', verbose=True)
# model.fit(train_pool)

# # 预测和评估
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')

# # 打印实际值与预测值
# for i in range(len(y_test)):
#     print(f"Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}")

#聚类hdbscan，
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from hdbscan import HDBSCAN

# 读取数据
data = pd.read_excel('河道氨氮统计数据--环境容量.xlsx', sheet_name='Sheet1')

# 假设目标变量是我们想要预测的氨氮浓度
y = data['氨氮浓度']

# 特征选择，这里我们选择除氨氮浓度和'RCH'外的数值型特征
X = data.drop(['氨氮浓度','RCH'], axis=1)

# 进行HDBSCAN聚类
clusterer = HDBSCAN(min_cluster_size=5)  # 可以根据数据调整min_cluster_size
X['cluster'] = clusterer.fit_predict(X)

# 将带有聚类标签的数据保存到CSV文件
X.to_csv('data_with_clusters.csv', index=False)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用Pool包装数据，以适应CatBoost的输入格式
train_pool = Pool(data=X_train, label=y_train)
test_pool = Pool(data=X_test, label=y_test)

# 初始化并训练模型
model = CatBoostRegressor(iterations=1000, learning_rate=0.1, loss_function='RMSE', verbose=True)
model.fit(train_pool)

# 预测和评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# 打印实际值与预测值
for i in range(len(y_test)):
    print(f"Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}")


#预测2，分类任务
# import pandas as pd
# from catboost import CatBoostClassifier, Pool
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # 加载数据
# data = pd.read_excel('河道氨氮统计数据--环境容量.xlsx', sheet_name='Sheet1')

# # 转换目标变量为分类标签
# # 假设我们将其分为两个类别：高浓度和低浓度
# threshold = data['氨氮浓度'].median()
# data['氨氮浓度_category'] = data['氨氮浓度'].apply(lambda x: 1 if x > threshold else 0)

# # 特征选择，这里我们选择除氨氮浓度和'RCH'外的数值型特征
# X = data.drop(['氨氮浓度', 'RCH', '氨氮浓度_category'], axis=1)
# y = data['氨氮浓度_category']

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 初始化分类器
# model = CatBoostClassifier(iterations=1000, learning_rate=0.1, loss_function='Logloss', verbose=True)

# # 使用Pool包装数据
# train_pool = Pool(data=X_train, label=y_train)
# test_pool = Pool(data=X_test, label=y_test)

# # 训练模型
# model.fit(train_pool)

# # 预测
# predictions = model.predict(X_test)

# # 计算准确率
# accuracy = accuracy_score(y_test, predictions)
# print(f'Accuracy: {accuracy * 100:.2f}%')

# # 输出预测结果与真实结果对比
# for i in range(len(y_test)):
#     print(f"Actual: {y_test.iloc[i]}, Predicted: {predictions[i]}")


