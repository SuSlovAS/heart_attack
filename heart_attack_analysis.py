import numpy as np
import pandas as pd
from warnings import filterwarnings
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from scipy.stats import shapiro, kstest
from scipy.stats import levene,bartlett
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_score,cross_val_predict


from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
filterwarnings('ignore')
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
for item in data_columns:
    print(data[item].value_counts(),'\n','-----' * 20)
print(data.groupby(['sex','output'])['trtbps'].mean())
print('-----' * 20)
print(data.groupby(['sex','output'])['chol'].mean())
print('-----' * 20)
print(data.groupby(['sex','output'])['thalachh'].mean())
print('-----' * 20)
print(data.groupby(['sex','output'])['oldpeak'].mean())
print('-----' * 20)

dataV = data.copy()
dataV['sex'] = pd.Categorical(dataV['sex'])
dataV['cp'] = pd.Categorical(dataV['cp'])
dataV['fbs'] = pd.Categorical(dataV['fbs'])
dataV['restecg'] = pd.Categorical(dataV['restecg'])
dataV['exng'] = pd.Categorical(dataV['exng'])
dataV['slp'] = pd.Categorical(dataV['slp'])
dataV['thall'] = pd.Categorical(dataV['thall'])
df = data.select_dtypes(include=['int32','float64','int64'])
#Searching correlation

corrPearson = data.corr(method='pearson')
corrSpearman = data.corr(method='spearman')

print('Pearson correlation\n',corrPearson)# Линейный коэффициент корреляции
print('Spearman correlation\n',corrSpearman)# Ранговый коэффициент корреляции Спирмана

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
#3D plot
fig_15 = plt.figure(figsize=(10,8))
ax_15 = Axes3D(fig_15)
ax_15.scatter(dataV['output'],dataV['trtbps'],dataV['chol'],c='green',
              s=30,alpha=0.5)
plt.show()
fig_16 = plt.figure(figsize=(10,8))
ax_16 = Axes3D(fig_16)
ax_16.scatter(dataV['output'],dataV['thalachh'],dataV['oldpeak'],c='blue',
              s=30,alpha=0.5)
plt.show()
fig_17 = plt.figure(figsize=(10,8))
ax_17 = Axes3D(fig_17)
ax_17.scatter(dataV['output'],dataV['chol'],dataV['thalachh'],c='red',
              s=30,alpha=0.5)
plt.show()
fig_18 = plt.figure(figsize=(10,8))
ax_18 = Axes3D(fig_18)
ax_18.scatter(dataV['output'],dataV['trtbps'],dataV['oldpeak'],c='yellow',
              s=30,alpha=0.5)
plt.show()
#Line plot
for item in data.columns[:-1]:
    fig = plt.figure(figsize=(10,8))
    ax = sns.lineplot(x='output',y=item,data=data)
    plt.show()
#Normality
data.hist(figsize=(20,8))
plt.show()
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
#Тест Левена (Если p < 0.05 значит нулевая гипотеза об однородности (гомоскедастичность) генаральной совокупности отклоняется)
print('statistic-%.4f-p_value-%.4f' % levene(data['age'],data['cp'],data['cp']
                                             ,data['trtbps'],data['chol'],
                                             data['fbs'],data['restecg'],
                                             data['thalachh'],data['exng'],
                                             data['oldpeak'],data['slp'],
                                             data['caa'],data['thall'],
                                             data['output']))
#Тест Баретлета (Если p < 0.05 значит нулевая гипотеза об однородности (гомоскедастичность) генаральной совокупности отклоняется)
print('statistic-%.4f-p_value-%.4f' % bartlett(data['age'],data['cp'],data['cp']
                                             ,data['trtbps'],data['chol'],
                                             data['fbs'],data['restecg'],
                                             data['thalachh'],data['exng'],
                                             data['oldpeak'],data['slp'],
                                             data['caa'],data['thall'],
                                             data['output']))
#Against Values
print('----' * 20)
data_against = data.copy()
clf = LocalOutlierFactor()
clf.fit_predict(data_against)
score = clf.negative_outlier_factor_
score_sorted = np.sort(score)
print(score_sorted[0:30])
point = score_sorted[12]
print(data_against[score == point])
against = data_against > point
print(data_against[against].isna().sum())
values = data_against < point
print(data_against[values].isna().sum())
print('----' * 20)

#Outliers
#One Class SVM (Если predict == -1, это выброс)
data_outliers = data.copy()
oc_svm = OneClassSVM(nu=0.25,gamma=0.5)
outliers_svm = data_outliers[oc_svm.fit_predict(data_outliers) == -1]
print(outliers_svm)
print('----' * 20)
#Isolation Forest
isfor = IsolationForest(contamination=0.01)
outliers_isfor = data_outliers[isfor.fit_predict(data_outliers) == -1]
print(outliers_isfor)
print('----' * 20)
#Local Outlier Factor
lof = LocalOutlierFactor()
lof.fit_predict(data_outliers)
outliers_lof = data_outliers[lof.fit_predict(data_outliers) == -1]
print(outliers_lof)
print('----' * 20)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print(scaled_data)
#Train test split
X , y = data.drop('output',axis=1,inplace=False), data['output']
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 train_size=0.75,
                                                 random_state=101,
                                                 stratify=y)
#Regression
lm = LinearRegression().fit(X_train,y_train)
pls = PLSRegression().fit(X_train,y_train)
knnr = KNeighborsRegressor().fit(X_train,y_train)
ridge = Ridge().fit(X_train,y_train)
lasso = Lasso().fit(X_train,y_train)
elasticnet = ElasticNet().fit(X_train,y_train)
cartr = RandomForestRegressor(random_state=42,verbose=False).fit(X_train,y_train)
baggr = BaggingRegressor(random_state=42,
                         bootstrap_features=True,
                         verbose=False).fit(X_train,y_train)
rfr = RandomForestRegressor(random_state=42,verbose=False).fit(X_train,y_train)
gbmr = GradientBoostingRegressor(verbose=False).fit(X_train,y_train)
xgbr = XGBRegressor().fit(X_train,y_train)
lgbm = LGBMRegressor().fit(X_train,y_train)
catbr = CatBoostRegressor(verbose=False).fit(X_train,y_train)
#Comparason
linear_models = [lm,pls,knnr,ridge,lasso,elasticnet,cartr,baggr,rfr,gbmr,
                 xgbr,lgbm,catbr]
for model in linear_models:
    name = model.__class__.__name__
    R2CV = cross_val_score(model,X_test,y_test,cv=10,scoring='r2').mean()
    error = -cross_val_score(model,X_test,y_test,cv=10,
                             scoring='neg_mean_squared_error').mean()
    print(name,':')
    print('----' * 20)
    print(R2CV)#чем ближе к 1 тем лучше модель (от 0 до 1)
    print(np.sqrt(error))
    print('----' * 20)
res = pd.DataFrame(columns=['models','r2cv'])
for model in linear_models:
    name = model.__class__.__name__
    r2cv = cross_val_score(model,X_test,y_test,cv=10,scoring='r2').mean()
    result = pd.DataFrame([[name,r2cv*100]],columns=['models','r2cv'])
    res = res.append(result)
fig_19 = plt.figure(figsize=(10,8))
ax_19 = sns.barplot(x='r2cv',y='models',data=res,color='k')
plt.xlabel('r2cv')
plt.ylabel('models')
plt.xlim(-50,100)
plt.title('Model accuracy comparasion')
plt.show()
#OLS
ols = sm.OLS(y_train,X_train).fit()
print(ols.summary())
#Standart Scaler
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.fit_transform(X_test)
y_train_scaled = np.array(y_train)
#PCA
pca = PCA()
X_Rtrain = pca.fit_transform(scale(X_train))
X_Rtest = pca.fit_transform(scale(X_test))

lmP = LinearRegression().fit(X_Rtrain,y_train)
R2CV = cross_val_score(lmP,X_Rtest,y_test,cv=10,scoring='r2').mean()
error = -cross_val_score(lmP,X_Rtest,y_test,cv=10,
                         scoring='neg_mean_squared_error').mean()
print('Linear Regression model with PCA')
print(R2CV)
print(np.sqrt(error))
print('----' * 20)
#MLPRegressor
mlpr = MLPRegressor().fit(X_train_scaled,y_train)
R2CV = cross_val_score(mlpr,X_test_scaled,y_test,cv=10,scoring='r2').mean()
error = -cross_val_score(mlpr,X_test_scaled,y_test,cv=10,
                         scoring='neg_mean_squared_error').mean()
print('MLP Regression model with standart scaler')
print(R2CV)
print(np.sqrt(error))
print('----' * 20)


#Classification
rfc = RandomForestClassifier(random_state=42,verbose=False).fit(X_train,y_train)
ctc = CatBoostClassifier(verbose=False).fit(X_train,y_train)
xgbc = XGBClassifier().fit(X_train,y_train)
lgbmc = LGBMClassifier().fit(X_train,y_train)
gbc = GradientBoostingClassifier(verbose=False).fit(X_train,y_train)
dtc = DecisionTreeClassifier(random_state=42).fit(X_train,y_train)
knnc = KNeighborsClassifier().fit(X_train,y_train)
svmc = SVC(verbose=False).fit(X_train,y_train)
lgr = LogisticRegression(solver='liblinear').fit(X_train,y_train)

#Comparason for classifiers
class_models = [rfc,ctc,xgbc,lgbmc,gbc,dtc,knnc,svmc,lgr]
for model in class_models:
    name = model.__class__.__name__
    predict = model.predict(X_test)
    R2CV = cross_val_score(model,X_test,y_test,cv=10,
                           verbose=False).mean()
    error = -cross_val_score(model,X_test,y_test,cv=10,
                             scoring ='neg_mean_squared_error',
                             verbose=False).mean()
    print(name,':')
    print('----' * 20)
    print(accuracy_score(y_test,predict))
    print(R2CV)
    print(np.sqrt(error))
    print('----' * 20)

res = pd.DataFrame(columns=['models','r2cv'])
for model in class_models:
    name = model.__class__.__name__
    r2cv = cross_val_score(model,X_test,y_test,cv=10,verbose=False).mean()
    result = pd.DataFrame([[name,r2cv*100]],columns=['models','r2cv'])
    res = res.append(result)
fig_20 = plt.figure(figsize=(10,8))
ax_20 = sns.barplot(x='r2cv',y='models',data=res,color='k')
plt.xlabel('r2cv')
plt.ylabel('models')
plt.xlim(0,100)
plt.title('Model accuracy comparasion')
plt.show()
