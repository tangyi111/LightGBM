# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
from lightgbm import plot_importance
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#对89950166、99999830这两个很接近的进行单独的训练预测

print('Loading data...')
train_data = pd.read_csv('train_org.csv')   # 读取数据
train_data = train_data.loc[train_data['current_service'].isin(['89950166','99999830'])]

train_data['service_type']=train_data['service_type'].astype(int)
train_data['is_mix_service']=train_data['is_mix_service'].astype(int)
train_data['online_time']=train_data['online_time'].astype(int)
train_data['many_over_bill']=train_data['many_over_bill'].astype(int)
train_data['contract_type']=train_data['contract_type'].astype(int)
train_data['contract_time']=train_data['contract_time'].astype(int)
train_data['is_promise_low_consume']=train_data['is_promise_low_consume'].astype(int)
train_data['net_service']=train_data['net_service'].astype(int)
train_data['pay_times']=train_data['pay_times'].astype(int)
train_data['gender']=train_data['gender'].astype(int)
train_data['age']=train_data['age'].astype(int)
train_data['complaint_level']=train_data['complaint_level'].astype(int)
train_data['former_complaint_num']=train_data['former_complaint_num'].astype(int)
train_data['current_service']=train_data['current_service'].astype(int)

dit = {89950166:0,  99999830:1}
antidit = {0: 89950166, 1: 99999830}

ty = train_data.pop('current_service')
ty = ty.astype(int)
y = [dit[x] for x in ty]

col = train_data.columns
x = train_data[col]
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.333, random_state=0)


train = lgb.Dataset(train_x,label=train_y)
valid = lgb.Dataset(valid_x, label=valid_y, reference=train)

#二分类参数
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary', #二分类
    'metric': 'auc',
    'num_leaves': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'seed':2018
}

print('Starting training...')
# 训练
gbm = lgb.train(params,
                train,
                num_boost_round=2000,
                valid_sets=valid,
                early_stopping_rounds=5)
plot_importance(gbm)
plt.show()
print('Saving model...')
# 保存模型
gbm.save_model('twoclasses.txt')

print('Starting predicting...')
# 预测
y_pred = gbm.predict(valid_x, num_iteration=gbm.best_iteration)
temp_result = [ int(y+0.5) for y in y_pred]

cnt1 = 0
cnt2 = 0
for i in range(len(valid_y)):
    if temp_result[i] == valid_y[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print(classification_report(valid_y,temp_result, digits=4))
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))


final_result = [antidit[x] for x in temp_result]
