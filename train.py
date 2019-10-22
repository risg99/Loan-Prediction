import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_og = train.copy()
test_og = test.copy()

# completing the missing values and outliers

# for categorical data replacing 'na' with the 'mode' value
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True) 
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

# for numerical data replacing 'na' with the most occuring value ie 360(occurs 512)
# print(train['Loan_Amount_Term'].value_counts()) returns 512
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

# for numerical data replacing 'na' with the median value
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


#repeating the same thing for test dataset
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) 
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

# removing the outliers from the train and test dataframes
train['LoanAmount_log'] = np.log(train['LoanAmount'])
test['LoanAmount_log'] = np.log(test['LoanAmount'])

# train['LoanAmount_log'].hist(bins=20)
# plt.show()

# as Loan_ID has no effect on the loan status, we drop them
train = train.drop('Loan_ID',axis=1) 
test = test.drop('Loan_ID',axis=1)

X = train.drop('Loan_Status',1) 
y = train.Loan_Status

X = pd.get_dummies(X)
train = pd.get_dummies(train) 
test = pd.get_dummies(test)

x_train, x_cv, y_train, y_cv = train_test_split(X,y,test_size=0.3)

model = LogisticRegression() 
model.fit(x_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=1, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
pred_cv = model.predict(x_cv)
pred_test = model.predict(test)
submission = pd.read_csv("Sample_Submission.csv")
submission['Loan_Status'] = pred_test 
submission['Loan_ID'] = test_og['Loan_ID']
submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)

# pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')