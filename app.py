import streamlit as st
import pickle
import pandas as pd
import base64
import tensorflow as tf
from tensorflow import keras
model = tf.keras.models.load_model('model.h5')
# Define the background image
background_image = '/home/deepin/Desktop/projects/hrv/WESAD/images.jpeg'
background_style = f"""
    <style>
    body {{
        background-image: url(data:image/jpeg;base64,{base64.b64encode(open(background_image, 'rb').read()).decode()});
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    </style>
"""

# Set up the Streamlit app
st.set_page_config(page_title='Stress Assessment', page_icon=':bar_chart:', layout='wide')

# Define colors for the app
primary_color = '#6c9ce4'
secondary_color = '#fc4f30'
text_color = 'white'

# Define the input fields
st.markdown(background_style, unsafe_allow_html=True)
st.title('Oxygen2innovate Stress Assessment')
st.write('Enter the following sensor data to predict stress level:')
with st.form(key='my_form'):
    col1, col2, col3 = st.columns(3)
    with col1:
        chestACCx = st.text_input('Chest Accelerometer X', key='chestACCx')
    with col2:
        chestACCy = st.text_input('Chest Accelerometer Y', key='chestACCy')
    with col3:
        chestACCz = st.text_input('Chest Accelerometer Z', key='chestACCz')
    with col1:
        chestEMG = st.text_input('Chest EMG', key='chestEMG')
    with col2:
        chestEDA = st.text_input('Chest EDA', key='chestEDA')
    with col3:
        chestTemp = st.text_input('Chest Temperature', key='chestTemp')
    with col1:
        submit_button = st.form_submit_button(label='Predict')

# Define the predict button
if submit_button:
    # Create a dictionary with the input data
    input_data = {
        'chestACCx': chestACCx,
        'chestACCy': chestACCy,
        'chestACCz': chestACCz,
        'chestEMG': chestEMG,
        'chestEDA': chestEDA,
        'chestTemp': chestTemp
    }

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
#     input_data = []

# for feature_name in feature_cols:
#     st.write(f"{'chestACCx'}:chestACCx,{'chestACCy'}:chestACCy,{'chestACCz'}:chestACCz,{'chestEMG'}:chestEMG,{'chestEDA'}:chestEDA,{'chestTemp'}:chestTemp")
#     value = st.number_input("", format="%.3f")
#     input_data.append(float(value))

# input_df = pd.DataFrame([input_data], columns=feature_cols)


    # Use the model to make a prediction
    prediction = model.predict(input_df)

    # Show the prediction label
    st.markdown('<hr>', unsafe_allow_html=True)
    st.write('## Prediction')
    st.write(f'The predicted stress level is: **{prediction[0]}**', unsafe_allow_html=True)

# Set the background color
st.markdown(f"""
    <style>
    .reportview-container {{
        background-color: transparent;
    }}   
    .sidebar .sidebar-content {{
        background-color: {primary_color};
    }}
    .streamlit-button {{
        background-color: {secondary_color};
        color: {text_color};
    }}
    </style>
""", unsafe_allow_html=True)
