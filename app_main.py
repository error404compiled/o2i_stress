# import streamlit as st
# import pandas as pd
# import numpy as np
# from tensorflow import keras

# # Load model
# model = keras.models.load_model('Main__\model.h5')

# # Define app layout
# st.set_page_config(page_title='Stress Prediction', page_icon=':heart:', layout='wide')
# st.title('Predicting Stress Levels Using Biometric Data')

# # Get user inputs
# st.header('Enter Biometric Data')
# chestACCx = st.number_input('Chest ACC x')
# chestACCy = st.number_input('Chest ACC y')
# chestACCz = st.number_input('Chest ACC z')
# chestECG = st.number_input('Chest ECG')
# chestEMG = st.number_input('Chest EMG')
# chestEDA = st.number_input('Chest EDA')
# chestTemp = st.number_input('Chest Temperature')
# chestResp = st.number_input('Chest Respiration Rate')
# height = st.number_input('Height (in cm)')
# weight = st.number_input('Weight (in kg)')

# # Define function to make prediction
# def make_prediction():
#     input_data = pd.DataFrame({
#         'chestACCx': [chestACCx],
#         'chestACCy': [chestACCy],
#         'chestACCz': [chestACCz],
#         'chestECG': [chestECG],
#         'chestEMG': [chestEMG],
#         'chestEDA': [chestEDA],
#         'chestTemp': [chestTemp],
#         'chestResp': [chestResp],
#         'height': [height],
#         'Weight': [weight]
#     })
#     y_pred = model.predict(input_data)
#     if np.argmax(y_pred[0]) == 0:
#         st.write("Result: Not Stressed")
#     else:
#         st.write("Result: Stressed")
#     st.write("Confidence Score: ", y_pred[0][np.argmax(y_pred[0])]*100)

# # Make prediction
# if st.button('Predict'):
#     make_prediction()

# import streamlit as st
# import pandas as pd
# import numpy as np
# from tensorflow import keras

# # Load model
# model = keras.models.load_model('Main__\model.h5')

# # Define input columns
# input_columns = ["chestACCx","chestACCy","chestACCz","chestECG","chestEMG","chestEDA","chestTemp","chestResp","height","Weight"]

# # Define function to get user inputs
# def get_user_inputs():
#     inputs = []
#     for col in input_columns:
#         val = st.number_input(f'{col} value:', step=0.1)
#         inputs.append(val)
#     return pd.DataFrame([inputs], columns=input_columns)

# # Define function to upload CSV file
# def upload_csv_file():
#     uploaded_file = st.file_uploader('Upload CSV file', type='csv')
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         if set(input_columns).issubset(set(df.columns)):
#             return df[input_columns].fillna(0)
#         else:
#             st.error(f'CSV file must contain columns {input_columns}')
#     return None

# # Define app layout
# st.set_page_config(page_title='Stress Prediction', page_icon=':heart:', layout='wide')
# st.title('Stress Prediction')

# # Get user inputs
# option = st.radio('Select input type:', ('Enter values', 'Upload CSV file'))

# st.subheader('Enter values in the table below:')
# if option == 'Enter values':
#     input_df = get_user_inputs()
# else:
#     input_df = upload_csv_file()

# if input_df is not None:
#     st.subheader('Input data:')
#     st.write(input_df)

# # Make prediction
# if input_df is not None:
#     if st.button('Predict'):
#         y_pred = model.predict(input_df)
#         if np.argmax(y_pred[0]) == 0:
#             st.write("Not Stressed")
#         else:
#             st.write("Stressed")
#         st.write("Confidence Score: ", y_pred[0][np.argmax(y_pred[0])]*100)



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
    input_data = {}
    for col in input_columns:
        input_data[col] = st.number_input(col, step=0.01)
    return pd.DataFrame(input_data, index=[0])

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

# Show input data in a table
if input_df is not None:
    st.subheader('Input Data')
    st.write(input_df)

# Make prediction
if input_df is not None:
    if st.button('Predict'):
        y_pred = model.predict(input_df)
        if np.argmax(y_pred[0]) == 0:
            st.write("Not Stressed")
        else:
            st.write("Stressed")
        st.write("Confidence Score: ", y_pred[0][np.argmax(y_pred[0])]*100)
