import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_og = train.copy()
test_og = test.copy()

#BIVARIATE ANALYSIS

#1. categorical data vs target variable

# Gender=pd.crosstab(train['Gender'],train['Loan_Status']) 
# Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
# plt.show()

# Married=pd.crosstab(train['Married'],train['Loan_Status']) 
# Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
# plt.show() 

# Dependents=pd.crosstab(train['Dependents'],train['Loan_Status']) 
# Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
# plt.show() 

# Education=pd.crosstab(train['Education'],train['Loan_Status']) 
# Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
# plt.show() 

# Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status']) 
# Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
# plt.show()

# Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status']) 
# Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
# plt.show() 

# Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])
# Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
# plt.show() 

#2. numerical data vs target variable

# train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
# or
# bins=[0,2500,4000,6000,81000] 
# group=['Low','Average','High', 'Very high'] 
# train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)
# Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status']) 
# Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
# plt.xlabel('ApplicantIncome') 
# plt.ylabel('Percentage')
# plt.show()

# bins=[0,1000,3000,42000] 
# group=['Low','Average','High'] 
# train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)
# Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status']) 
# Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
# plt.xlabel('CoapplicantIncome') 
# plt.ylabel('Percentage')
# plt.show()

# target variable doesnt seem much dependent on applicant income
# but is more dependent on coapplicant income which is wrong
# since most of them donot have coapplicants, hence their value
# must have been zero; hence the reason we are creating a new variable
# which combines both the incomes

# train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
# bins=[0,2500,4000,6000,81000] 
# group=['Low','Average','High','Very high'] 
# train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
# Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status']) 
# Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
# plt.xlabel('Total_Income') 
# plt.ylabel('Percentage')
# plt.show()

# bins=[0,100,200,700] 
# group=['Low','Average','High'] 
# train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
# LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
# LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
# plt.xlabel('LoanAmount') 
# plt.ylabel('Percentage')
# plt.show()

# plotting sns for seeing which variable is most likely
# to approve laon status



