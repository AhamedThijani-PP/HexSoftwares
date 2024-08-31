import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv('heart.csv')

data=pd.get_dummies(data,columns=['cp','restecg','slope','ca','thal'])

scaler=StandardScaler()
data[['age','trestbps','chol','thalach','oldpeak']]=scaler.fit_transform(data[['age','trestbps','chol','thalach','oldpeak']])

for col in data.columns:
    if data[col].dtype == bool:
        data[col]=data[col].astype(int)

X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model=RandomForestClassifier()
model.fit(X_train_scaled,y_train)

y_pred=model.predict(X_test_scaled)

acc=accuracy_score(y_test,y_pred)
print("Accuracy : ",acc)