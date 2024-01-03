import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
from geopy.distance import great_circle

# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide
st.set_page_config(page_title="Trip Duration Predictor", page_icon="🚕", layout="wide")
st.markdown("<h1 style='text-align: center; color : #E8570E; font-size:65px'>New York City Trip Duration Analyser</h1>", unsafe_allow_html=True)
st.markdown("<br><hr><br>", unsafe_allow_html=True)


TrainData = pd.read_csv("Dataset/TrainData.csv")
TestData = pd.read_csv("Dataset/TestData.csv")

TrainData['pickup_date'] = pd.to_datetime(pd.to_datetime(TrainData['pickup_datetime']).dt.date)
st.subheader('Train Trips over Time')
fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
ax1.plot(TrainData.groupby('pickup_date').count()[['vendor_id']], 'o-', label='train', alpha=0.7)
ax1.set_title('Train Trips over Time')
ax1.legend(loc='upper left')
ax1.set_ylabel('Trips')
ax1.set_xlabel('Date')
st.pyplot(fig)


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
st.subheader('Distribution of Pickup Locations - Train vs. Test')
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 7))
axes[0].scatter(TrainData['pickup_longitude'], TrainData['pickup_latitude'], s=2, label='Train', alpha=0.1)
axes[1].scatter(TestData['pickup_longitude'], TestData['pickup_latitude'], color='orangered', s=2, label='Test', alpha=0.1)

fig.suptitle('Distribution of Pickup Locations - Train vs. Test')
axes[0].set_title('Train Data')
axes[1].set_title('Test Data')
axes[0].set_ylabel('Latitude')
axes[0].set_xlabel('Longitude')
axes[1].set_xlabel('Longitude')
axes[0].legend(loc='upper left')
axes[1].legend(loc='upper left')
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
st.pyplot(fig)


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
st.subheader('Distribution of Dropoff Locations - Train vs. Test')
train_downsampled = TrainData.sample(frac=0.1)
test_downsampled = TestData.sample(frac=0.1)
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 7))
axes[0].scatter(train_downsampled['dropoff_longitude'], train_downsampled['dropoff_latitude'], s=2, label='Train', alpha=0.1)
axes[0].set_title('Train Data')
axes[0].set_ylabel('Latitude')
axes[0].set_xlabel('Longitude')
axes[0].legend(loc='upper left')
axes[0].set_ylim(city_lat_border)
axes[0].set_xlim(city_long_border)
axes[1].scatter(test_downsampled['dropoff_longitude'], test_downsampled['dropoff_latitude'], color='orangered', s=2, label='Test', alpha=0.1)
axes[1].set_title('Test Data')
axes[1].set_xlabel('Longitude')
axes[1].legend(loc='upper left')
axes[1].set_ylim(city_lat_border)
axes[1].set_xlim(city_long_border)
st.pyplot(fig)


st.subheader('Distribution of Trip Duration')
fig, ax = plt.subplots(1, 1, figsize=(15, 4))
hist1 = sns.histplot(data=TrainData, x=np.log1p(TrainData['trip_duration']), ax=ax, kde=True)
hist1.set_title('Distribution of Trip Duration')
hist1.set_xlabel('Log(Trip Duration + 1)') 
st.pyplot(fig)


st.subheader('Distribution of Latitude and Longitudes')
df = TrainData.loc[(TrainData.pickup_latitude > 40.5) & (TrainData.pickup_latitude < 41)]
df = df.loc[(df.dropoff_latitude > 40.5) & (df.dropoff_latitude < 41)]
df = df.loc[(df.dropoff_longitude > -74.05) & (df.dropoff_longitude < -73.7)]
df = df.loc[(df.pickup_longitude > -74.05) & (df.pickup_longitude < -73.7)]
sns.set(style="white", palette="muted", color_codes=True)
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=False, sharey=False)
sns.despine(left=True)
sns.distplot(df['pickup_latitude'].values, label='pickup_latitude', color="m", bins=100, ax=axes[0, 0])
sns.distplot(df['pickup_longitude'].values, label='pickup_longitude', color="g", bins=100, ax=axes[0, 1])
sns.distplot(df['dropoff_latitude'].values, label='dropoff_latitude', color="m", bins=100, ax=axes[1, 0])
sns.distplot(df['dropoff_longitude'].values, label='dropoff_longitude', color="g", bins=100, ax=axes[1, 1])
plt.setp(axes, yticks=[])
plt.tight_layout()
st.pyplot(fig)


st.subheader('Count Plot of Vendor IDs')
fig, ax = plt.subplots(figsize=(10, 6))
sns.set_style("whitegrid")
plot = sns.countplot(data=TrainData, x='vendor_id', palette='Set2')
for p in plot.patches:
    plot.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
plt.xlabel('Vendor ID')
plt.ylabel('Count')
plt.title('Count Plot of Vendor IDs')
sns.despine(trim=True)
plt.xticks(rotation=0)
st.pyplot(fig)