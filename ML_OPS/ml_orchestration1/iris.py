import streamlit as st
import joblib
import numpy as np

# scaler = load(open('models/standard_scaler.pkl', 'rb'))
pipeline = joblib.load(open(rf"model.pkl", 'rb'))
target_names = ['setosa', 'versicolor', 'virginica']
out_dict = {}
for i, target in enumerate(target_names):
    out_dict[i] = target

sl = st.text_input("Sepal Length", placeholder="Enter value in cm")
sw = st.text_input("Sepal Width", placeholder="Enter value in cm")
pl = st.text_input("Petal Length", placeholder="Enter value in cm")
pw = st.text_input("Petal Width", placeholder="Enter value in cm")

btn_click = st.button("Predict")

if btn_click == True:
    if sl and sw and pl and pw:
        query_point = np.array([float(sl), float(sw), float(pl), float(pw)]).reshape(1, -1)
        # query_point_transformed = scaler.transform(query_point)
        pred = pipeline.predict(query_point)[0]
        st.success(out_dict[pred])
    else:
        st.error("Enter the values properly.")