import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras

# Load model
model = keras.models.load_model('Main__\model.h5')

# Define input columns
input_columns = ["chestACCx","chestACCy","chestACCz","chestECG","chestEMG","chestEDA","chestTemp","chestResp","height","Weight"]

# Define function to get user inputs
def get_user_inputs():
    inputs = []
    for col in input_columns:
        val = st.text_input(f'Enter {col} value:')
        try:
            val = float(val)
        except:
            val = np.nan
        inputs.append(val)
    return pd.DataFrame([inputs], columns=input_columns)

# Define function to upload CSV file
def upload_csv_file():
    uploaded_file = st.file_uploader('Upload CSV file', type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if set(input_columns).issubset(set(df.columns)):
            return df[input_columns].fillna(0)
        else:
            st.error(f'CSV file must contain columns {input_columns}')
    return None

# Define app layout
st.set_page_config(page_title='Stress Prediction', page_icon=':heart:', layout='wide')
st.title('Stress Prediction')

# Get user inputs
option = st.radio('Select input type:', ('Enter values', 'Upload CSV file'))
if option == 'Enter values':
    input_df = get_user_inputs()
else:
    input_df = upload_csv_file()

# Make prediction
if input_df is not None:
    if st.button('Predict'):
        y_pred = model.predict(input_df)
        if np.argmax(y_pred[0]) == 0:
            st.write("Not Stressed")
        else:
            st.write("Stressed")
        st.write("Confidence Score: ", y_pred[0][np.argmax(y_pred[0])]*100)