import streamlit as st

import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Trip Duration Predictor", page_icon="🚕", layout="wide")
st.markdown("<h1 style='text-align: center; color : #E8570E; font-size:65px'>New York City Trip Duration Predictor</h1>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

TrainData = pd.read_csv("Dataset/TrainDataFS.csv")
def remove_outliers(data, column_name):
    Q1 = np.percentile(data[column_name], 25)
    Q3 = np.percentile(data[column_name], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    outliers_removed = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    return outliers_removed

TrainData=remove_outliers(TrainData, 'pickup_longitude')
TrainData=remove_outliers(TrainData, 'pickup_latitude')
TrainData=remove_outliers(TrainData, 'trip_duration')
X = TrainData.drop(columns=['trip_duration'])
y = TrainData['trip_duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tabs1, tabs2, tabs3 = st.tabs(['Introduction', 'Instructions', 'Example Input'])

with tabs1:
    st.write("""
             
             ## Predict Trip Duration

            Welcome to the Predict Trip Duration page! Use the form on the left to input details for a taxi trip, 
            and our trained model will predict the trip duration for you. This page allows you to interactively explore 
            how the model performs on custom input data.""")

with tabs2:
    st.write("""
             
             ### Instructions:

            1. **Prediction:** Give the input and see the model's predicted trip duration.
             """)
    
with tabs3:
    st.write("""### Example Input:

        - Vendor ID: 1
        - Pickup Hour: 12
        - Pickup Longitude: -73.95
        - Pickup Latitude: 40.75
        - Dropoff Longitude: -73.95
        - Dropoff Latitude: 40.75
        - Passenger Count: 1
        - Pickup Datetime: YYYY-MM-DD HH:MM:SS

        Feel free to experiment with different inputs and observe how the model responds. If the predicted and actual values are close, it indicates a successful prediction.""")
    
st.markdown("<br><hr><br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
vendor_id = col1.radio("Select Vendor ID", [1, 2], index=0)
pickup_hour = col2.slider("Select Pickup Hour", 0, 23, 12)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
pickup_longitude = col1.number_input("Enter Pickup Longitude", value=-73.95000, step=0.0001, min_value=-74.5, max_value=-73.50, format="%.8f")
pickup_latitude = col2.number_input("Enter Pickup Latitude", value=40.75000, step=0.0001, min_value=40.50, max_value=40.90, format="%.8f")
dropoff_longitude = col1.number_input("Enter Dropoff Longitude", value=-73.95000, step=0.0001, min_value=-74.5, max_value=-73.50, format="%.8f")
dropoff_latitude = col2.number_input("Enter Dropoff Latitude", value=40.75000, step=0.0001, min_value=40.50, max_value=40.90, format="%.8f")

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
passenger_count = col1.number_input("Enter Passenger Count", value=1)
pickup_date = col2.date_input("PickupDate", datetime.date(2016, 1, 1), min_value=datetime.date(2016, 1, 1), max_value=datetime.date(2016, 12, 31))

st.markdown("<br><hr><br>", unsafe_allow_html=True)

pickup_month = pickup_date.month
pickup_weekday = pickup_date.weekday()

def haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

distance = haversine_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)

#vendor_id, passenger_count, pickup_longitude, pickup_latitude, pickup_weekday, pickup_hour, pickup_month, distance

with open('Models/best_estimator_mlp.pkl', 'rb') as model_file:
    modelDa = pickle.load(model_file)

client_predictions = {}
client_models = {}

for client_id in range(5):
    model_filename = f"Models/FL_Client {client_id}.pkl"
    with open(model_filename, 'rb') as model_file:
        local_model = pickle.load(model_file)
    client_models[client_id] = local_model

for client_id in range(5):
    local_model = client_models[client_id]    
    client_pred = local_model.predict(X_test)
    client_predictions[client_id] = client_pred

noise_level=0.2
for client_id in range(5):
    for j in range(len(client_predictions[0])):
        noise = np.random.uniform(-noise_level, noise_level)
        client_predictions[client_id][j]=client_predictions[client_id][j]+noise

prediction = np.mean([client_predictions[client_id] for client_id in range(5)], axis=0)

rmsle_value = np.sqrt(np.mean(np.square(np.subtract(np.log1p(y_test), np.log1p(prediction)))))

input_data = np.array([[vendor_id, passenger_count, pickup_longitude, pickup_latitude, pickup_weekday, pickup_hour, pickup_month, distance]])
for client_id in range(5):
    local_model = client_models[client_id]    
    client_pred = local_model.predict(input_data)
    client_predictions[client_id] = client_pred
noise_level=0.2
for client_id in range(5):
    for j in range(len(client_predictions[0])):
        noise = np.random.uniform(-noise_level, noise_level)
        client_predictions[client_id][j]=client_predictions[client_id][j]+noise
prediction = np.mean([client_predictions[client_id] for client_id in range(5)], axis=0)

predicted_duration_seconds = round(prediction[0], 2)
predicted_minutes = int(predicted_duration_seconds // 60)
predicted_seconds = int(predicted_duration_seconds % 60)

st.markdown(f"<p style='font-size:30px; color:#FFFFFF;'>Predicted Trip Duration: {predicted_minutes} minutes and {predicted_seconds} seconds</p>", unsafe_allow_html=True)
for client_id in range(5):
    st.write("Client ", client_id, " : ", round(client_predictions[client_id][0], 2))
data = {
    'Vendor ID': vendor_id,
    'Passenger Count': passenger_count,
    'Pickup Longitude': pickup_longitude,
    'Pickup Latitude': pickup_latitude,
    'Pickup Weekday': pickup_weekday,
    'Pickup Hour': pickup_hour,
    'Pickup Month': pickup_month,
    'Distance': round(distance, 2)
}
df = pd.DataFrame([data])
st.write(df)