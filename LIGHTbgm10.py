import lightgbm as lgb
import pandas as pd
from lightgbm import plot_importance
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import f1_score

print('Loading data...')
train_data = pd.read_csv('train_org.csv')   # 读取训练集数据

#转化数据类型
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


#套餐的分类(10)
dit = {89950166:0, 89950167:1, 89950168:2, 90063345:3, 90109916:4, 90155946:5, 99999825:6, 99999826:7, 99999827:8, 99999828:9, 99999830:0} #有两个套餐接近，打上同一标签
antidit = {0: 89950166, 1: 89950167, 2: 89950168, 3: 90063345, 4: 90109916, 5: 90155946, 6: 99999825, 7: 99999826, 8: 99999827, 9: 99999828, 10: 99999830} #给的11个套餐

ty = train_data.pop('current_service')
ty = ty.astype(int)
y = [dit[x] for x in ty]

col = train_data.columns
x = train_data[col]
# 完整模板：
# train_X,test_X,train_y,test_y = train_test_split(train_data,train_target,test_size=0.3,random_state=5)
# 参数解释：
# train_data：待划分样本数据
# train_target：待划分样本数据的结果（标签）
# test_size：测试数据占样本数据的比例，若整数则样本数量
# random_state：设置随机数种子，保证每次都是同一个随机数。若为0或不填，则每次得到数据都不一样
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.333, random_state=0)  #train_test_split()函数划分训练、测试数据

#数据转换
train = lgb.Dataset(train_x,label=train_y)
valid = lgb.Dataset(valid_x, label=valid_y, reference=train)  #有效 valid

#，设置参数，多分类(10)
params = {
    'boosting_type': 'gbdt',  #训练方式 梯度提升
    'objective': 'multiclass', #目标  多分类
    'metric': 'multi_logloss',  #损失函数
    'num_class': 10,  #10个类
    'num_leaves': 1000,  #叶子节点数，树模型复杂度
    'learning_rate': 0.05,  #学习率  0.01~0.3
    'feature_fraction': 0.9,  #每次迭代中随机选择特征的比例 ，加速训练，处理过拟合
    'bagging_fraction': 0.8,  #不进行重采样的情况下随机选择部分数据，加速训练，处理过拟合
    'bagging_freq': 5,  #bdgging的次数
    'verbose': 0,
    'seed':2018
}


print('Starting training...')
#训练
gbm = lgb.train(params,  #参数字典
                train,   #训练集
                num_boost_round=2000,   #迭代次数
                valid_sets=valid,       #验证集
                early_stopping_rounds=5)    #早停系数

print('Saving model...')
# 保存模型
gbm.save_model('modewait.txt')

print('Starting predicting...')
#预测
y_pred = gbm.predict(valid_x, num_iteration=gbm.best_iteration)
temp_result = [y.argmax() for y in y_pred]

cnt1 = 0   #模型预测和样本分出的测试集 一致
cnt2 = 0   #模型预测和样本分出的测试集 不一致
for i in range(len(valid_y)):
    if temp_result[i] == valid_y[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print(classification_report(valid_y,temp_result, digits=4))  #fscore
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))  #模型准确性




