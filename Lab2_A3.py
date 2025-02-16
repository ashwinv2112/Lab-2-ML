import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_excel("C:/Users/ashwi/Desktop/Sem 4/ML/Lab_ML/Lab Session Data.xlsx", sheet_name="Purchase data",usecols="A:E")
data["Category"]=data["Payment (Rs)"].apply(lambda x: "RICH" if x>200 else "POOR")

x=data.drop(columns=["Payment (Rs)","Customer","Category"])
y=data["Category"]

x=pd.get_dummies(pd.DataFrame(x),drop_first=True)
xtrain, xtest, ytrain, ytest=train_test_split(x,y,test_size=0.3,random_state=42)

scale=StandardScaler()
xtrain=scale.fit_transform(xtrain)
xtest=scale.transform(xtest)
model=LogisticRegression(class_weight="balanced")
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)

accuracy=accuracy_score(ytest,ypred)
print("Accuracy- ",accuracy*100)

