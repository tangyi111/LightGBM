import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot
import matplotlib as mpl
import seaborn as sns

#表   首先对几列定量数据进行分析，得到结果如下
data = pd.read_csv('train_org.csv', usecols=['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee'])
print(data.describe())


#多彩图片  各个数据的分布情况  为方便对数据进行离散化，统计了数据的分布情况，绘制图表如下所示
data = pd.read_csv('train_org.csv')

sns.countplot(x='complaint_level',hue='current_service',data=data)
plt.show()
sns.countplot(x='gender',hue='current_service',data=data)
plt.show()
sns.countplot(x='is_promise_low_consume',hue='current_service',data=data)
plt.show()
sns.countplot(x='many_over_bill',hue='current_service',data=data)
plt.show()
sns.countplot(x='service_type',hue='current_service',data=data)
plt.show()

