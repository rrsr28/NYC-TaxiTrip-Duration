# New York City Trip Duration Predictor

## Overview

Welcome to the New York City Trip Duration Predictor project! This project focuses on predicting the duration of taxi trips in the bustling city of New York. Leveraging the power of machine learning, specifically a Multi-Layer Perceptron (MLP), we aim to build a model that accurately estimates the time it takes for a taxi journey from pickup to dropoff.

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
3. **Prediction:** Use the trained model to make predictions on new taxi trip data.

## Dependencies

- [Streamlit](https://www.streamlit.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Altair](https://altair-viz.github.io/)
- [Plotly Express](https://plotly.com/python/)
