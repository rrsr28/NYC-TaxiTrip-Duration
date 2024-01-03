import streamlit as st

# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide

st.set_page_config(page_title="Trip Duration Predictor", page_icon="🚕", layout="wide")
st.markdown("<h1 style='text-align: center; color : #E8570E; font-size:65px'>New York City Trip Duration Predictor</h1>", unsafe_allow_html=True)
st.markdown("<br><hr><br>", unsafe_allow_html=True)

st.write(
        """
        ## Overview

        Welcome to the New York City Taxi Trip Duration Prediction project! This project focuses on predicting the duration of taxi trips in the bustling city of New York. Leveraging the power of machine learning, specifically a Multi-Layer Perceptron (MLP), we aim to build a model that accurately estimates the time it takes for a taxi journey from pickup to dropoff.

        ## Dataset

        The dataset used for this project contains a rich set of features, including vendor information, pickup and dropoff timestamps, geographical coordinates, passenger count, and more. Exploring this dataset provides valuable insights into the patterns and factors influencing taxi trip durations.

        ## Approach

        Our approach involves the implementation of a Multi-Layer Perceptron, a type of artificial neural network, for regression tasks. By training the model on historical taxi trip data, we aim to capture the intricate relationships between various features and trip durations, enabling accurate predictions for new trips.

        ## Key Features

        - **MLP Regression:** Utilizing a Multi-Layer Perceptron for regression tasks.
        - **Geospatial Analysis:** Exploring geographical features such as pickup/dropoff coordinates.

        ## Usage

        1. **Data Preprocessing:** Before running the MLP model, preprocess the dataset to handle missing values, convert timestamps, and normalize features.
        2. **Model Training:** Train the MLP regression model using the processed dataset.
        3. **Prediction:** Trained model to make predictions on new taxi trip data.

        """
    )