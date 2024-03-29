import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. 读取数据
data = pd.read_csv('dataset.csv')

# 2. 数据预处理
# 使用min-max方法将输入数据的所有特征值归一化到0-6的范围内
X = data.iloc[:, 5:-1]  # 特征值
y = data.iloc[:, -1]  # 目标值
scaler = MinMaxScaler(feature_range=(0, 6))
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集，比例为85:15
X_train, X_test, y_train, y_test = (train_test_split(X_scaled, y, test_size=0.15, random_state=21))

# 4. 使用GPR模型训练，设置超参数
kernel = RationalQuadratic(alpha=1, length_scale=1)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=0, random_state=0)
gpr.fit(X_train, y_train)

# 5. 预测
y_pred_train = gpr.predict(X_train)
y_pred_test = gpr.predict(X_test)


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
