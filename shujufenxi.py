import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot
import matplotlib as mpl

#age
data = pd.read_csv('train_org.csv')

component_col = data['age']
component_categories = component_col.unique()

categories_grouped = data.groupby('age')
categories_count = categories_grouped['age'].count()
print(categories_count)

categories_count.plot(kind='bar')
plt.title('age')
plt.tight_layout()
plt.show()

#1_total_fee
pr = pd.read_csv("train_org.csv")
print("统计年龄分布：")
print()
age = []
for i in pr['1_total_fee']:
    age.append(i)
print("1_total_fee")
age1=[]
age2=[]
age3=[]
age4=[]
age5=[]
for i in age:
    if 0<=i<100:
        age1.append(i)
    elif 100<=i<200:
        age2.append(i)
    elif 200<=i<300:
        age3.append(i)
    elif 300<=i<400:
        age4.append(i)
    else:
        age5.append(i)

index=['0~100','100~200','200~300','300~400','others']
values=[len(age1),len(age2),len(age3),len(age4),len(age5)]
plt.title('4_total_fee')
plt.bar(index,values)
plt.show()

#2_total_fee
pr = pd.read_csv("train_org.csv")
print("统计年龄分布：")
print()
age = []
for i in pr['2_total_fee']:
    age.append(i)
print("1_total_fee")
age1=[]
age2=[]
age3=[]
age4=[]
age5=[]
for i in age:
    if 0<=i<100:
        age1.append(i)
    elif 100<=i<200:
        age2.append(i)
    elif 200<=i<300:
        age3.append(i)
    elif 300<=i<400:
        age4.append(i)
    else:
        age5.append(i)

index=['0~100','100~200','200~300','300~400','others']
values=[len(age1),len(age2),len(age3),len(age4),len(age5)]
plt.title('4_total_fee')
plt.bar(index,values)
plt.show()

#3_total_fee
pr = pd.read_csv("train_org.csv")
print("统计年龄分布：")
print()
age = []
for i in pr['3_total_fee']:
    age.append(i)
print("1_total_fee")
age1=[]
age2=[]
age3=[]
age4=[]
age5=[]
for i in age:
    if 0<=i<100:
        age1.append(i)
    elif 100<=i<200:
        age2.append(i)
    elif 200<=i<300:
        age3.append(i)
    elif 300<=i<400:
        age4.append(i)
    else:
        age5.append(i)

index=['0~100','100~200','200~300','300~400','others']
values=[len(age1),len(age2),len(age3),len(age4),len(age5)]
plt.title('4_total_fee')
plt.bar(index,values)
plt.show()

#4_total_fee
pr = pd.read_csv("train_org.csv")
print("统计年龄分布：")
print()
age = []
for i in pr['4_total_fee']:
    age.append(i)
print("1_total_fee")
age1=[]
age2=[]
age3=[]
age4=[]
age5=[]
for i in age:
    if 0<=i<100:
        age1.append(i)
    elif 100<=i<200:
        age2.append(i)
    elif 200<=i<300:
        age3.append(i)
    elif 300<=i<400:
        age4.append(i)
    else:
        age5.append(i)

index=['0~100','100~200','200~300','300~400','others']
values=[len(age1),len(age2),len(age3),len(age4),len(age5)]
plt.title('4_total_fee')
plt.bar(index,values)
plt.show()
