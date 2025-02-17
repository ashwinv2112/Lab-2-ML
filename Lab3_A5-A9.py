from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

data=pd.read_csv("C:/Users/ashwi/Desktop/Sem 4/ML/Lab_ML/Lab-3/accident.csv")

encoder = LabelEncoder()
data['Gender']=encoder.fit_transform(data['Gender'])
data['Helmet_Used']=encoder.fit_transform(data['Helmet_Used'])
data['Seatbelt_Used']=encoder.fit_transform(data['Seatbelt_Used'])

impute=SimpleImputer(strategy='mean')
data[['Age','Speed_of_Impact','Gender','Helmet_Used','Seatbelt_Used']]=impute.fit_transform(data[['Age','Speed_of_Impact','Gender','Helmet_Used','Seatbelt_Used']])

#A4
x=data[['Age','Gender','Speed_of_Impact','Helmet_Used','Seatbelt_Used']]
y=data['Survived']
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=42)

#A5
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)

#A6
# Test the accuracy using the test set (X_test, y_test)
accuracy = neigh.score(x_test, y_test)
print(f"Accuracy- {accuracy * 100:.2f}%")

#A7
predictions = neigh.predict(x_test)
print("Predictions for the test set- ", predictions)

#A8
k=range(1, 12)
accuracies = []

# Loop through each k value
for i in k:
    neigh2=KNeighborsClassifier(n_neighbors=i)
    neigh2.fit(x_train, y_train)
    predictions2=neigh2.predict(x_test)

    accuracy2=accuracy_score(y_test, predictions2)
    accuracies.append(accuracy2)
    print(f"Accuracy for {i} neighbor(s)- {accuracy2 * 100:.2f}%")

plt.plot(k, accuracies, marker='o', linestyle='-', color='b')
plt.title('kNN Classifier Accuracy for Different k Values')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.xticks(k)
plt.grid(True)
plt.show()

#A9
predictions3 = neigh.predict(x_test)

CM=confusion_matrix(y_test, predictions3)
plt.figure(figsize=(6, 5))
sns.heatmap(CM, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix for Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

prec=precision_score(y_test, predictions3)
recall=recall_score(y_test, predictions3)
f1=f1_score(y_test, predictions3)

print(f"Precision- {prec:.2f}")
print(f"Recall- {recall:.2f}")
print(f"F1-Score- {f1:.2f}")

train_pred=neigh.predict(x_train)

train_CM=confusion_matrix(y_train, train_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(train_CM, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix for Training Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

train_prec=precision_score(y_train, train_pred)
train_recall=recall_score(y_train, train_pred)
train_f1=f1_score(y_train, train_pred)

print(f"Training Precision- {train_prec:.2f}")
print(f"Training Recall- {train_recall:.2f}")
print(f"Training F1-Score- {train_f1:.2f}")