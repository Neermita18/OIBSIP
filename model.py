import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
df= pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv?raw=True")
print(df)
print(df.describe())

X= df.iloc[:,0:4]
print(X)
y = df['species'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# with open('scaler.pkl', 'wb') as file:
#     pkl.dump(scaler, file)
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
model2= GradientBoostingClassifier(n_estimators=100, random_state=42)
model2.fit(X_train, y_train)
y_pred2=model2.predict(X_test)
model3= DecisionTreeClassifier(criterion="entropy", max_depth=8, min_samples_leaf=1, random_state=42)
model3.fit(X_train, y_train)
y_pred3= model3.predict(X_test)

accuracy = accuracy_score(y_test, y_pred3)
print(f'Accuracy: {accuracy:.2f}')


print('Classification Report:')
print(classification_report(y_test, y_pred3))

print('Confusion Matrix:')
cm=(confusion_matrix(y_test, y_pred3))

plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model3.classes_, yticklabels=model3.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

with open('decision_tree.pkl', 'wb') as file:
    pkl.dump(model3, file)