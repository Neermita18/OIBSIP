import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px

df= pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv?raw=True")

with open('random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)
with open('decision_tree.pkl', 'rb') as file:
    dt_model = pickle.load(file)
with open('gradient_boost.pkl', 'rb') as file:
    gb_model = pickle.load(file)
    
# Create interactive feature importance plot
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fig = px.bar(df, x='Feature', y='Importance', title= model.__class__.__name__)
    st.plotly_chart(fig)

st.cache_data.clear()

st.sidebar.header("Input flower features")
sepal_length = st.sidebar.slider("Sepal Length", min_value=4.0, max_value=8.0, value=5.0, step=0.1)
sepal_width = st.sidebar.slider("Sepal Width", min_value=1.0, max_value=4.5, value=3.0, step=0.1)
petal_length = st.sidebar.slider("Petal Length", min_value=1.0, max_value=7.0, value=4.0, step=0.1)
petal_width = st.sidebar.slider("Petal Width", min_value=0.1, max_value=2.5, value=1.2, step=0.1)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

scaled_features = scaler.transform(features)

st.title("Predict flower species using various Machine Learning models")
st.image("all.png")
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
    prediction = rf_model.predict(scaled_features)
    st.header(f"Random Forest Prediction: {prediction[0]}")
    if prediction[0]=="Iris-setosa":
        st.image("setosa.jpg")
    if prediction[0]=="Iris-versicolor":
        st.image("Iris_versicolor_3.jpg")
    if prediction[0]=="Iris-virginica":
        st.image("iris_virginica.jpg")
    
    st.text("Random Forest Feature Importance")
    plot_feature_importance(rf_model, df.columns[0:4])
if dt_but:
    prediction = dt_model.predict(scaled_features)
    st.header(f"Decision Tree Prediction: {prediction[0]}")
    if prediction[0]=="Iris-setosa":
        st.image("setosa.jpg")
    if prediction[0]=="Iris-versicolor":
        st.image("Iris_versicolor_3.jpg")
    if prediction[0]=="Iris-virginica":
        st.image("iris_virginica.jpg")
    st.text("Decision Tree Feature Importance")
    plot_feature_importance(dt_model, df.columns[0:4])
if gb_but:
    prediction = gb_model.predict(scaled_features)
    st.header(f"Gradient Boosting Prediction: {prediction[0]}")
    if prediction[0]=="Iris-setosa":
        st.image("setosa.jpg")
    if prediction[0]=="Iris-versicolor":
        st.image("Iris_versicolor_3.jpg")
    if prediction[0]=="Iris-virginica":
        st.image("iris_virginica.jpg")
    st.text("Gradient Boosting Feature Importance")
    plot_feature_importance(gb_model, df.columns[0:4])
    
    

