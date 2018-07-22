# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 23:46:24 2018

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:24:34 2018

@author: Chexki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%  Data Loading

loan_df= pd.read_csv(r'D:\Python_Project\XYZCorp_LendingData.txt',header=0,delimiter='\t',low_memory=False)
#print(loan_df)

#%%
# Generating a dummy dataset

loan_df1 = pd.DataFrame.copy(loan_df)
print(loan_df1.describe(include= 'all'))

#%% Feature Selection

# Some of the variables are not helpful in order to build a predictive model, hence dropping.

loan_df1.drop(['id','member_id','funded_amnt_inv','grade','emp_title','pymnt_plan','desc','title','addr_state',
            'inq_last_6mths','mths_since_last_record','initial_list_status','mths_since_last_major_derog','policy_code','application_type'
            ,'annual_inc_joint','dti_joint','verification_status_joint','tot_coll_amt','tot_cur_bal','open_acc_6m','open_il_6m','open_il_12m'
            ,'open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m',
            'max_bal_bc','all_util','inq_fi','total_cu_tl','inq_last_12m'],axis=1,inplace=True)

print(loan_df1.head())

#%%
# Checking if missing values are present.
loan_df1.isnull().sum()

#%%
print(loan_df1.dtypes)

#%%
# Imputing categorical missing data with mode value

colname1=['term','sub_grade','emp_length','home_ownership','verification_status',
          'issue_d','purpose','zip_code','earliest_cr_line','last_pymnt_d',
          'next_pymnt_d','last_credit_pull_d']
for x in colname1[:]:
    loan_df1[x].fillna(loan_df1[x].mode()[0],inplace=True)
#%%
loan_df1.isnull().sum()

#%%
colname2=['mths_since_last_delinq','revol_util','collections_12_mths_ex_med',
          'total_rev_hi_lim']
for x in colname2[:]:
    loan_df1[x].fillna(loan_df1[x].mean(),inplace=True)
    
loan_df1.isnull().sum()

#%%
print(loan_df1.shape)
loan_df1.describe()    

########################################################################################
#%%         OUTLIERS
#%%   Exploratory analysis : Graphical representation to know more about data,
#     Using boxplot.

loan_df1.boxplot()
#%%
plt.xticks(rotation=90) 
plt.show()

#%%
# Clearly outliers are present.
# Using Capping and Flooring Method
# Handling Outliers by replacing the with mean of respective features. 

#%%
loan_df1.boxplot()
plt.xticks(rotation=90) 
plt.show()

#%%
pd.set_option("display.max_columns",None)

#%%
# Label Encoding
colname1=['term','sub_grade','emp_length','home_ownership','verification_status',
          'purpose','zip_code','earliest_cr_line','last_pymnt_d',
          'next_pymnt_d','last_credit_pull_d']

from sklearn import preprocessing
le={}                                          # create blank dictionary

for x in colname1:
    le[x]=preprocessing.LabelEncoder()  
    
# create labels (1,2,3,4) to different categories in variables
# assigning numerical levels
    
for x in colname1:
    loan_df1[x]= le[x].fit_transform(loan_df1.__getattr__(x))
    
loan_df1.head()
    
#%%
# Defining Dependent (Y) and Independent (X) variables
 
#X = loan_df1.values[:,:-1]          #independent vars
#Y = loan_df1.values[:,-1]           # dependent var
#print(Y)

#%%
# Splitting  the data into testing and training

##from sklearn.model_selection import train_test_split
##X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3,
                      #                              random_state=10)



#%%
test_list=["Jun-2015","Jul-2015","Aug-2015","Sep-2015","Oct-2015","Nov-2015","Dec-2015"]
test_list
#%%
oot_test = loan_df1.loc[loan_df1.issue_d.isin(test_list)]
oot_test.shape
#oot_test.head()

#%%
train = loan_df1.loc[-loan_df1.issue_d.isin(test_list)]
train.shape

#%%
oot_test=oot_test.drop("issue_d",axis=1)
train=train.drop("issue_d",axis=1)

#%%

# Training a Logistic Regression Model.
X_test = oot_test.values[:,:-1]
Y_test= oot_test.values[:,-1]

#print(X_test,Y_test)
#%%
X_train = train.values[:,:-1]
Y_train= train.values[:,-1]


#%%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
#print(x)
scaler.fit(X_test)
X_test=scaler.transform(X_test)

#%%
from sklearn.linear_model import LogisticRegression

# create a model
classifier = (LogisticRegression())

# fitting training data to the model

classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

#%%
print(list(zip(Y_test,Y_pred)))

print(Y_test)
#%%
print(Y_pred)

#%%

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cfm= confusion_matrix(Y_test,Y_pred)

print(cfm)

print("Classification Report")
print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("Accuracy of the Model:", acc)

# 0.980

#%%

# Adjusting The Threshold

# stores the predicted probabilities

y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)

#%%   
y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value < 0.35:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
#print(y_pred_class)

#%%

from sklearn.metrics import confusion_matrix, accuracy_score

cfm= confusion_matrix(Y_test.tolist(),y_pred_class)

print(cfm)

accuracy_score = accuracy_score(Y_test.tolist(),y_pred_class)
print("Accuracy of the Model:", accuracy_score)
    
#%%

# Using cross Validation

classifier=(LogisticRegression())

from sklearn import cross_validation

# Performing kfold cross validation

kfold_cv = cross_validation.KFold(n=len(X_train),n_folds = 10)
print(kfold_cv)

# running the model using scoring metric accuracy

kfold_cv_result = cross_validation.cross_val_score( \
                                        estimator =classifier,
                                        X=X_train,y=Y_train,
                                        scoring="accuracy",
                                        cv=kfold_cv)
print(kfold_cv_result)
#[ 0.98240929  0.98105745  0.98149137  0.98304349  0.98154144  0.98095731
#  0.98092393  0.98152475  0.98150775  0.98152444]

# finding the mean
print(kfold_cv_result.mean())
# 0.981598

#%%