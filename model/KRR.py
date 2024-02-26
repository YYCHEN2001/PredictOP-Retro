import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# 1. 读取数据
data = pd.read_csv('dataset.csv')

# 2. 设置前五列为索引（此处不使用作为特征），最后一列为目标值，其余为特征值
X = data.iloc[:, 5:-1]  # 特征值
y = data.iloc[:, -1]  # 目标值
scaler = MinMaxScaler(feature_range=(0, 6))
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

# 4. 使用KRR模型训练，设置超参数alpha=0.1, coef0=4, degree=2, kernel='poly'
krr = KernelRidge(alpha=0.1, kernel='poly', degree=2, coef0=4)
krr.fit(X_train, y_train)

# 5. 预测
y_pred_train = krr.predict(X_train)
y_pred_test = krr.predict(X_test)

# 6. 计算评估指标
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, mape, rmse


metrics_train = calculate_metrics(y_train, y_pred_train)
metrics_test = calculate_metrics(y_test, y_pred_test)

# 7. 以表格形式输出训练集和测试集的四种评估指标
results = pd.DataFrame({
    'Metric': ['R2', 'MAE', 'MAPE', 'RMSE'],
    'Train': metrics_train,
    'Test': metrics_test
})

print(results)
