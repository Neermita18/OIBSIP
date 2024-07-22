import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px
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
st.cache()
df= pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv?raw=True")


X= df.iloc[:,0:4]

y = df['species'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# with open('scaler.pkl', 'wb') as file:
#     pkl.dump(scaler, file)







df= pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv?raw=True")


    
# Create interactive feature importance plot
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fig = px.bar(df, x='Feature', y='Importance', title= model.__class__.__name__)
    st.plotly_chart(fig)



st.sidebar.header("Input flower features")
sepal_length = st.sidebar.slider("Sepal Length", min_value=4.0, max_value=8.0, value=5.0, step=0.1)
sepal_width = st.sidebar.slider("Sepal Width", min_value=1.0, max_value=4.5, value=3.0, step=0.1)
petal_length = st.sidebar.slider("Petal Length", min_value=1.0, max_value=7.0, value=4.0, step=0.1)
petal_width = st.sidebar.slider("Petal Width", min_value=0.1, max_value=2.5, value=1.2, step=0.1)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])


scaled_features = scaler.transform(features)

st.title("Predict flower species using various Machine Learning models")
st.image("assets/all.png")
st.text("<- Use the sidebar to specify your flower's features")
col1, col2, col3 = st.columns(3)
with col1.container(border=True, height=200):
    st.header("Random Forest")
    rf_but=st.button("Use Random Forest to predict", key="rf")
    
with col2.container(border=True, height=200):
    st.header("Decision Tree")
    dt_but=st.button("Use Decision Tree to predict", key="dt")
with col3.container(border=True, height=200):
    st.header("Gradient Boosting")
    gb_but=st.button("Use Gradient Boosting to predict", key="gb")
    
if rf_but:
    model1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model1.fit(X_train, y_train)
    prediction = model1.predict(scaled_features)
    st.header(f"Random Forest Prediction: {prediction[0]}")
    if prediction[0]=="Iris-setosa":
        st.image("assets/setosa.jpg")
    if prediction[0]=="Iris-versicolor":
        st.image("assets/Iris_versicolor_3.jpg")
    if prediction[0]=="Iris-virginica":
        st.image("assests/iris_virginica.jpg")
    
    st.text("Random Forest Feature Importance")
    plot_feature_importance(model1, df.columns[0:4])
if dt_but:
    model3= DecisionTreeClassifier(criterion="entropy", max_depth=8, min_samples_leaf=1, random_state=42)
    model3.fit(X_train, y_train)
    prediction = model3.predict(scaled_features)
    st.header(f"Decision Tree Prediction: {prediction[0]}")
    if prediction[0]=="Iris-setosa":
        st.image("assets/setosa.jpg")
    if prediction[0]=="Iris-versicolor":
        st.image("assets/Iris_versicolor_3.jpg")
    if prediction[0]=="Iris-virginica":
        st.image("assests/iris_virginica.jpg")
    plot_feature_importance(model3, df.columns[0:4])
if gb_but:
    model2= GradientBoostingClassifier(n_estimators=100, random_state=42)
    model2.fit(X_train, y_train)
    prediction = model2.predict(scaled_features)
    st.header(f"Gradient Boosting Prediction: {prediction[0]}")
    if prediction[0]=="Iris-setosa":
        st.image("assets/setosa.jpg")
    if prediction[0]=="Iris-versicolor":
        st.image("assets/Iris_versicolor_3.jpg")
    if prediction[0]=="Iris-virginica":
        st.image("assests/iris_virginica.jpg")
    st.text("Gradient Boosting Feature Importance")
    plot_feature_importance(model2, df.columns[0:4])
    
    

