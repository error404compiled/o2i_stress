import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

st.title("Emotion Recognition App")

st.write("Please provide the following details:")

chestACCx = st.number_input("Chest ACC x:",format="%.20f", step=1e-10)
chestACCy = st.number_input("Chest ACC y:", format="%.20f",step=1e-10)
chestACCz = st.number_input("Chest ACC z:", format="%.20f",step=1e-10)
chestEMG = st.number_input("Chest EMG:", format="%.20f",step=1e-10)
chestEDA = st.number_input("Chest EDA:", format="%.20f",step=1e-10)
chestTemp = st.number_input("Chest Temperature:",format="%.20f", step=1e-10)

if st.button("Predict"):
    input_data = np.array([chestACCx, chestACCy, chestACCz, chestEMG, chestEDA, chestTemp]).reshape(1,-1)

    # Load the trained model
    model = load_model("model.h5")

    # Make predictions on the input data
    scaler = StandardScaler()
    X_test = scaler.fit_transform(input_data)
    y_pred = model.predict(X_test)

    if np.argmax(y_pred[0]) == 0:
        st.write("Normal")
    elif np.argmax(y_pred[0]) == 1:
        st.write("Stressed")
    else:
        st.write("Amused")
