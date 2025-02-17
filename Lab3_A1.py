import numpy as np
import pandas as pd

data=pd.read_csv("C:/Users/ashwi/Desktop/Sem 4/ML/Lab_ML/Lab-3/accident.csv")

data.replace(
    {
        'Gender':{'Male':0,'Female':1},
        'Helmet_Used':{'No':0,'Yes':1},
        'Seatbelt_Used': {'No': 0, 'Yes': 1}

    }, inplace=True)
f=['Age', 'Gender', 'Speed_of_Impact', 'Helmet_Used', 'Seatbelt_Used']

class1=data[data["Survived"] == 0][f]
class2=data[data["Survived"] == 1][f]

centroid1=np.mean(class1, axis=0)
centroid2=np.mean(class2, axis=0)

stddev1=np.std(class1, axis=0)
stddev2=np.std(class2, axis=0)

interclass=np.linalg.norm(centroid1-centroid2)

print("Centroid for class 1 (Not Survived)-\n", centroid1)
print("----------------------------------------------")
print("Centroid for class 2 (Survived)-\n", centroid2)

print("\nStandard Deviation for class 1 (Not Survived)-\n", stddev1)
print("----------------------------------------------")
print("Standard Deviation for class 2 (Survived)- \n", stddev2)

print(f"\nInterclass Distance- {interclass:.2f}")
