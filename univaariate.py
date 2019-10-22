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


# UNIVARIATE ANALYSIS:
#1. plotting categorical data
# plt.subplot(221) 
# train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 
# plt.subplot(222) 
# train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
# plt.subplot(223) 
# train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
# plt.subplot(224) 
# train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 
# plt.show()

#2. plotting cardinal data
# plt.subplot(131) 
# train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents') 
# plt.subplot(132) 
# train['Education'].value_counts(normalize=True).plot.bar(title= 'Education') 
# plt.subplot(133) 
# train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area') 
# plt.show()

#3. plotting numerical data
# plt.subplot(121) 
# sns.distplot(train['ApplicantIncome'])
# plt.subplot(122) 
# train['ApplicantIncome'].plot.box(figsize=(16,5)) 
# train.boxplot(column='ApplicantIncome', by = 'Education') 
# plt.suptitle("")
# plt.show()

# plt.subplot(121) 
# sns.distplot(train['CoapplicantIncome']); 
# plt.subplot(122) 
# train['CoapplicantIncome'].plot.box(figsize=(16,5)) 
# plt.show()

# plt.subplot(121) 
# df=train.dropna() 
# sns.distplot(df['LoanAmount']) 
# plt.subplot(122) 
# train['LoanAmount'].plot.box(figsize=(16,5)) 
# plt.show()
