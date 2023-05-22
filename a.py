# importing libraries start ---
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier #the classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# importing libraries end---
# .
# .
# .
# .
# .
# loading data start---
test_data=pd.read_csv('test.csv')
train_data=pd.read_csv('train.csv')
# loading data end---
# .
# .
# .
# .
# .
# show data start---
# train_data['age'].plot(marker='.', linestyle='')
# plt.show()
# show data end---
# .
# .
# .
# .
# .
# cleaning data start---
# ... droping nulls and removing the id irrelevant column :-
train_data.drop('id',axis=1,inplace=True)
# ------------------- making an id copy for the testing data
test_data['id']=test_data['id'].round().astype(int)
temp_id=pd.DataFrame( test_data['id'].copy())
test_data.drop('id',axis=1,inplace=True)
# ...
# ...
# ...
print(train_data.info())
# remove outliers
# bmi
# Q1 = train_data['bmi'].quantile(0.25)
# Q3 = train_data['bmi'].quantile(0.75)
# IQR = Q3 - Q1
# lower = Q1 - 1.5*IQR
# upper = Q3 + 1.5*IQR
# upper_array = np.where(train_data['bmi']>=upper)[0]
# lower_array = np.where(train_data['bmi']<=lower)[0]
# train_data.drop(index=upper_array, inplace=True)
# train_data.drop(index=lower_array, inplace=True)
# fill nulls with mean
test_data['bmi'].fillna((test_data['bmi'].mean()), inplace=True)
# filling cetegorial null with mode
test_data['smoking_status'].replace({'Unknown':np.nan}, inplace=True) 
test_data['smoking_status'].fillna(test_data['smoking_status'].mode()[0], inplace=True)
# ...  transforming training data values into numbers :-
train_data['gender'].replace({'Female':1, 'Male':0,'Other':np.nan}, inplace=True)
train_data['work_type'].replace({'Private':0, 'Self-employed':1,  'Govt_job':2, 'Never_worked':3, 'children':4}, inplace=True)
train_data['ever_married'].replace({'Yes':1, 'No':0}, inplace=True)
train_data['Residence_type'].replace({'Rural':1, 'Urban':0}, inplace=True)
train_data['smoking_status'].replace({'formerly smoked':1, 'never smoked':0, 'smokes':2, 'Unknown':np.nan}, inplace=True) 
# ...  transforming testing data values into numbers :-
test_data['smoking_status'].replace({'formerly smoked':1, 'never smoked':0, 'smokes':2, 'Unknown':np.nan}, inplace=True)
test_data['gender'].replace({'Female':1, 'Male':0}, inplace=True)
test_data['work_type'].replace({'Private':0, 'Self-employed':1,  'Govt_job':2, 'Never_worked':3, 'children':4}, inplace=True)
test_data['ever_married'].replace({'Yes':1, 'No':0}, inplace=True)
test_data['Residence_type'].replace({'Rural':1, 'Urban':0}, inplace=True)
# drop_the_nulls
train_data.dropna(inplace=True)
print(train_data.info())
# making better weights
drop_num=int(len(train_data[train_data['stroke']==0])-len(train_data[train_data['stroke']==1]))
train_data.drop(train_data[train_data['stroke']==0].tail(drop_num).index,inplace=True)
# .
# .
# .
# .
# .
# predicting the stroke start---
# ...  making an x and y variables for splitting purposes :-
X = train_data.drop("stroke", axis=1)
y = train_data["stroke"]
# ...
# ...
# ...
# ...  splitting the data :-
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.9)
CLF =SVC()
CLF = CLF.fit(x_train, y_train)
# ...
# ...
# ...
# ... predicting
y_predict = CLF.predict(test_data)
result_df=temp_id
result_df["stroke"]=y_predict
result_df['stroke'] = result_df['stroke'].astype('int')
    # print(result_df)
    # print(column_counts)

result_df.to_csv('predictieddd.csv', header=True,index=False) # putting the results in a file


# result_df.to_csv('predictieddd.csv', header=True,index=True) # putting the results in a file
# ...
# ...
# ...
# ... testing accuracy
# predicting the stroke end---
# .
# .
# .
# .
# .
# printing results start---
# printing results end---
# .
# .
# .
# .
# .




