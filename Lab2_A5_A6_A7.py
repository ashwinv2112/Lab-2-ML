import pandas as pd
file="C:/Users/ashwi/Desktop/Sem 4/ML/Lab_ML/Lab Session Data.xlsx"
data=pd.read_excel(file, sheet_name="thyroid0387_UCI")

print("Data Types-")
print(data.dtypes)

#2
categorical = data.select_dtypes(include=['object']).columns
numerical = data.select_dtypes(include=['int64', 'float64']).columns
print("\nCategorical Columns-", categorical.tolist())
print("Numerical Columns-", numerical.tolist())
print("---------------------")
numerical = numerical.drop('Record ID')

#3
print("\nData Range-")
print(data[numerical].describe())
print("---------------------")

#4
print("Missing Values-")
print(data.isnull().sum())
print("---------------------")

#5
Q1=data["age"].quantile(0.25)
Q3=data["age"].quantile(0.75)
IQR=Q3-Q1
lower=Q1-1.5*IQR
upper=Q3+1.5*IQR
outliers=data[(data["age"]<lower)|(data["age"]>upper)]
print("Outliers- \n", outliers)
print("---------------------")

#6
#A6- SINCE OUR DATASET HAS OUTLIERS, WE'RE FILLING IT WITH MEDIAN
med=data["age"].median()
data["age"]=data["age"].apply(lambda i: med if i < lower or i > upper else i)

print("Mean- ")
print(data[numerical].mean())
print("\nVariance- ")
print(data[numerical].var())
print("\nStandard Deviation- ")
print(data[numerical].std())
print("---------------------")

#A7
cols=['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
data[cols]=data[cols].apply(pd.to_numeric, errors='coerce')

data[['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']] = (data[['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']]-data[['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']].mean())/data[['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']].std()
print(data[['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']].mean())
print(data[['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']].std())

