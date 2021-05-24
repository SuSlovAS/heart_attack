import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import shapiro, kstest
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
import seaborn as sns

heart_data = pd.read_csv('heart.csv')
col_cp = {0:'typical angina',1:'atypical angina',2:'non-anginal pain',3:'asymptomatic'}
data = heart_data.copy()

print(data.head(20))
print(data.isna().sum())
print(data.tail(10))
print(data.info())
print(data.describe())
print(data.shape)
print(data.corr())
data_columns = data.columns.to_list()


dataV = data.copy()
#Searching correlation

corrPearson = data.corr(method='pearson')
corrSpearman = data.corr(method='spearman')

#print('Pearson correlation\n',corrPearson)# Линейный коэффициент корреляции
#print('Spearman correlation\n',corrSpearman)# Ранговый коэффициент корреляции Спирмана

fig_1 = plt.figure(figsize=(10,8))
ax_1 = sns.heatmap(corrPearson,annot=True,vmin=-1,vmax=+1,
                   yticklabels=True,cbar=True,cmap='viridis')
sns.set_style('whitegrid')
plt.title('Pearson')
plt.xlabel('Columns')
plt.ylabel('Columns')
plt.show()

fig_2 = plt.figure(figsize=(10,8))
ax_2 = sns.heatmap(corrSpearman,annot=True,vmin=-1,vmax=+1,
                   yticklabels=True,cbar=True,cmap='viridis')
sns.set_style('whitegrid')
plt.title('Spearman')
plt.xlabel('Columns')
plt.ylabel('Columns')
plt.show()

fig_data = data.hist(figsize=(20,10))
plt.show()

#Box plot
fig_3 = plt.figure(figsize=(10,7))
ax_3 = sns.boxplot(x='trtbps',y='output',data=data[['trtbps','output']],
                   orient = 'h',palette='Set3')
plt.show()

fig_4 = plt.figure(figsize=(10,7))
ax_4 = sns.boxplot(x='chol',y='output',data=data[['chol','output']],
                   orient='h',palette='Set3')
plt.show()

fig_5 = plt.figure(figsize=(10,7))
ax_5 = sns.boxplot(x='thalachh',y='output',data=data[['thalachh','output']],
                   orient='h',palette='Set3')
plt.show()

fig_6 = plt.figure(figsize=(10,7))
ax_6 = sns.boxplot(x='oldpeak',y='output',data=data[['oldpeak','output']],
                  orient='h',palette='Set3')
plt.show()

fig_7 = plt.figure(figsize=(10,7))
ax_7 = sns.boxplot(x='age',y='output',data=data[['age','output']],
                   orient='h',palette='Set3')
plt.show()

#Bar plot
fig_8 = plt.figure(figsize=(10,8))
ax_8 = sns.barplot(x='sex',y='output',data=data[['sex','output']],
                   palette='magma')
plt.show()
fig_9 = plt.figure(figsize=(10,8))
ax_9 = sns.barplot(x='cp',y='output',data=data[['cp','output']],
                   palette='magma')
plt.show()
fig_10 = plt.figure(figsize=(10,8))
ax_10 = sns.barplot(x='fbs',y='output',data=data[['fbs','output']],
                    palette='magma')
plt.show()
fig_11 = plt.figure(figsize=(10,8))
ax_11 = sns.barplot(x='restecg',y='output',data=data[['restecg','output']],
                    palette='magma')
plt.show()
fig_12 = plt.figure(figsize=(10,8))
ax_12 = sns.barplot(x='exng',y='output',data=data[['exng','output']],
                    palette='magma')
plt.show()
fig_13 = plt.figure(figsize=(10,8))
ax_13 = sns.barplot(x='slp',y='output',data=data[['slp','output']],
                    palette='magma')
plt.show()
fig_14 = plt.figure(figsize=(10,8))
ax_14 = sns.barplot(x='thall',y='output',data=data[['thall','output']],
                    palette='magma')
plt.show()
#Normality

#Shapiro-Wilk test 0 hyposthisis if pvalue<0.05, then distribution is not normal(Gaussian)
print('Shapiro-Wilk test')
for i in data.columns:
    print('-----'*10)
    shapiro_test = shapiro(data[i])
    print('%.3f - %.3f' % shapiro_test)
#Kolmagorov-Smirnov test 0 hyposthisis if pvalue<0.05, then distribution is not like compared distribution 
print('Kolmagorov-Smirnov test')
for i in data.columns:
    print('-----'*10)
    kolmagorov_test = kstest(data[i], 'norm')
    print('%.3f - %.3f' % kolmagorov_test)
#Homogenety (Однородность)


scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)
print(scaled_data)











