
try:
    from autogluon.tabular import TabularPredictor, TabularDataset
except ImportError:
    !pip install autogluon
    from autogluon.tabular import TabularPredictor, TabularDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========================================================
# 第一部分：固定基礎區 (資料載入)
# ========================================================
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_sub = pd.read_csv('sample_submission.csv')

df_train['target'] = np.log1p(df_train['result'])
train_data = df_train.drop(columns=['result', 'id'])
test_data = df_test.drop(columns=['id'])

# ========================================================
# 第二部分：實驗調參區 (AutoGluon 自動建模)
# ========================================================


# 定義訓練參數
# presets: 'best_quality' 會自動執行 Bagging 與 Stacking
save_path = 'ag_models_v6'
predictor = TabularPredictor(
    label='target',
    problem_type='regression',
    eval_metric='rmse',
    path=save_path
).fit(
    train_data,
    presets='best_quality',
    time_limit=600,
    num_bag_folds=5,
    num_stack_levels=1
)

# 查看模型排行榜
leaderboard = predictor.leaderboard(train_data, silent=False)
print(leaderboard)

# 獲取預測結果
pred_ensemble = predictor.predict(test_data)
final_log_pred = pred_ensemble

# ========================================================
# 第三部分：固定輸出區 (逆轉換與生成檔案)
# ========================================================

# 修改版本編號
version_code = "v6"

# 逆對數轉換 (Expm1) 並進行 [0, 1] 約束
final_result = np.expm1(final_log_pred)
final_result = np.clip(final_result, 0, 1)

# 儲存檔案
df_sub['result'] = final_result
output_name = f'submission_optimized_{version_code}.csv'
df_sub.to_csv(output_name, index=False)

print(f"已產出檔案：{output_name}")
print(f"預測結果平均值：{final_result.mean():.6f}")