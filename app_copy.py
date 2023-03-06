import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Define the feature column names and their data types
feature_cols = ['chestACCx', 'chestACCy', 'chestACCz', 'chestEMG', 'chestEDA', 'chestTemp']
dtypes = {'chestACCx': 'float64', 'chestACCy': 'float64', 'chestACCz': 'float64', 'chestEMG': 'float64', 'chestEDA': 'float64', 'chestTemp': 'float64'}

# Define a function to preprocess the input data
def preprocess_input(input_df):
    # Fill missing values with zero
    input_df.fillna(0, inplace=True)
    # Convert the data types of the feature columns
    for col in feature_cols:
        input_df[col] = input_df[col].astype(dtypes[col])
    # Return the preprocessed input dataframe
    return input_df[feature_cols]

# Define the app title and sidebar title
st.set_page_config(page_title="HRV Predictor", page_icon=":heart:", layout="wide")
st.sidebar.title("HRV Predictor")

# Define a function to get user inputs
def get_user_inputs():
    # Initialize a dictionary to store the user inputs
    user_input = {}
    # Loop through the feature columns and get user inputs
    for col in feature_cols:
        user_input[col] = st.sidebar.number_input(col, value=0, format='%.2f')
    # Convert the dictionary to a dataframe and return it
    return pd.DataFrame(user_input, index=[0])

# Define the app content
def app():
    # Set the app title
    st.title("HRV Predictor")
    # Get the user inputs
    input_df = get_user_inputs()
    # Preprocess the input data
    input_df_processed = preprocess_input(input_df)
    # Make a prediction
    prediction = model.predict(input_df_processed)
    # Display the prediction
    st.subheader("HRV Prediction")
    st.write("The predicted HRV value is", prediction[0][0])

# Run the app
if __name__ == "__main__":
    app()
