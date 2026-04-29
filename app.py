import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Data load karo
dataset = pd.read_csv("D:/diabetes.csv")

X = dataset.drop(['Outcome'], axis=1)
y = dataset['Outcome']

X = (X - np.min(X)) / (np.max(X) - np.min(X))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# UI
st.title("🩺 Diabetes Prediction App")
st.write("Patient ka data daalo aur result dekho!")

pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 300)
bp = st.number_input("Blood Pressure", 0, 200)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 0, 120)

if st.button("Predict!"):
    new_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    new_data = (new_data - np.min(X.values)) / (np.max(X.values) - np.min(X.values))
    result = knn.predict(new_data)
    
    if result[0] == 1:
        st.error("⚠️ Diabetes ka risk hai!")
    else:
        st.success("✅ Diabetes ka risk nahi hai!")