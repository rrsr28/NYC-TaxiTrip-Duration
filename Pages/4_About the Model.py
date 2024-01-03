import streamlit as st

import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide

st.set_page_config(page_title="Trip Duration Predictor", page_icon="🚕", layout="wide")
st.markdown("<h1 style='text-align: center; color : #E8570E; font-size:65px'>About the Model</h1>", unsafe_allow_html=True)
st.markdown("<br><hr><br>", unsafe_allow_html=True)

st.title("🌟 MLPRegressor Model Overview 🌟")

st.markdown("""
#### Overview:
The MLPRegressor (Multi-Layer Perceptron Regressor) is a powerful neural network model used for regression tasks. It's known for its ability to approximate complex non-linear functions.

#### Model Architecture:
- **Hidden Layers**: This model has two hidden layers, each containing 9 neurons.
- **Learning Rate**: The initial learning rate is set to 0.1.
- **Maximum Iterations**: The model will be trained for a maximum of 1000 iterations.

""")

st.markdown("""
#### Key Parameters:
- **Hidden Layer Sizes**: (9, 9)
- **Learning Rate Init**: 0.1
- **Max Iterations**: 1000

""")

st.markdown("""
#### Use Cases:
- MLPRegressor is suitable for various regression problems, including predicting housing prices, stock market trends, and more.
- Its flexibility in modeling non-linear relationships makes it an excellent choice for complex data patterns.

#### Tips:
- Experiment with different hyperparameters to fine-tune your model's performance.
- Consider using early stopping to prevent overfitting during training.
""")

st.markdown("""<br><hr><br>""", unsafe_allow_html=True)

st.subheader("Search CV Results")

st.markdown(
        """<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_warm_start</th>
      <th>param_max_iter</th>
      <th>param_learning_rate_init</th>
      <th>param_hidden_layer_sizes</th>
      <th>param_early_stopping</th>
      <th>param_activation</th>
      <th>...</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>split5_test_score</th>
      <th>split6_test_score</th>
      <th>split7_test_score</th>
      <th>split8_test_score</th>
      <th>split9_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>67.589050</td>
      <td>25.123150</td>
      <td>0.070870</td>
      <td>0.024237</td>
      <td>True</td>
      <td>1000</td>
      <td>0.1</td>
      <td>(9, 9)</td>
      <td>False</td>
      <td>relu</td>
      <td>...</td>
      <td>-0.456403</td>
      <td>-0.490559</td>
      <td>-0.490746</td>
      <td>-0.489737</td>
      <td>-0.480618</td>
      <td>-0.483114</td>
      <td>-0.462077</td>
      <td>-0.484115</td>
      <td>0.020191</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>41.815025</td>
      <td>11.577858</td>
      <td>0.051881</td>
      <td>0.046545</td>
      <td>True</td>
      <td>1000</td>
      <td>0.1</td>
      <td>(4,)</td>
      <td>False</td>
      <td>relu</td>
      <td>...</td>
      <td>-0.508846</td>
      <td>-0.479420</td>
      <td>-0.551639</td>
      <td>-0.474222</td>
      <td>-0.525323</td>
      <td>-0.468287</td>
      <td>-0.486474</td>
      <td>-0.497993</td>
      <td>0.025376</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>39.600636</td>
      <td>12.139525</td>
      <td>0.081757</td>
      <td>0.051575</td>
      <td>True</td>
      <td>1000</td>
      <td>0.3</td>
      <td>(7,)</td>
      <td>False</td>
      <td>relu</td>
      <td>...</td>
      <td>-0.510058</td>
      <td>-0.538729</td>
      <td>-0.531465</td>
      <td>-0.558886</td>
      <td>-0.510506</td>
      <td>-0.492314</td>
      <td>-0.522124</td>
      <td>-0.522829</td>
      <td>0.025087</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>44.762769</td>
      <td>12.568637</td>
      <td>0.072754</td>
      <td>0.026179</td>
      <td>True</td>
      <td>1000</td>
      <td>0.5</td>
      <td>(5, 5)</td>
      <td>False</td>
      <td>relu</td>
      <td>...</td>
      <td>-0.501208</td>
      <td>-0.503702</td>
      <td>-0.744472</td>
      <td>-0.504535</td>
      <td>-0.525061</td>
      <td>-0.501533</td>
      <td>-0.528368</td>
      <td>-0.537842</td>
      <td>0.070043</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>24.395271</td>
      <td>6.727169</td>
      <td>0.053628</td>
      <td>0.027435</td>
      <td>True</td>
      <td>1000</td>
      <td>1</td>
      <td>(4,)</td>
      <td>False</td>
      <td>relu</td>
      <td>...</td>
      <td>-0.551504</td>
      <td>-0.511053</td>
      <td>-0.567137</td>
      <td>-0.509782</td>
      <td>-0.736894</td>
      <td>-0.555086</td>
      <td>-0.735342</td>
      <td>-0.592798</td>
      <td>0.096180</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22.245912</td>
      <td>6.040664</td>
      <td>0.053825</td>
      <td>0.021777</td>
      <td>True</td>
      <td>1000</td>
      <td>1</td>
      <td>(5,)</td>
      <td>False</td>
      <td>relu</td>
      <td>...</td>
      <td>-0.497529</td>
      <td>-0.734355</td>
      <td>-0.735515</td>
      <td>-0.737610</td>
      <td>-0.736962</td>
      <td>-0.570396</td>
      <td>-0.734864</td>
      <td>-0.672457</td>
      <td>0.098058</td>
      <td>6</td>
    </tr>
    <tr>
      <th>0</th>
      <td>22.758047</td>
      <td>9.747841</td>
      <td>0.074132</td>
      <td>0.026028</td>
      <td>True</td>
      <td>1000</td>
      <td>2</td>
      <td>(6, 6, 6)</td>
      <td>False</td>
      <td>relu</td>
      <td>...</td>
      <td>-0.735016</td>
      <td>-0.737287</td>
      <td>-0.737844</td>
      <td>-0.733357</td>
      <td>-0.734220</td>
      <td>-0.732251</td>
      <td>-0.732649</td>
      <td>-0.735537</td>
      <td>0.002280</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>43.348169</td>
      <td>13.646893</td>
      <td>0.078411</td>
      <td>0.039613</td>
      <td>True</td>
      <td>1000</td>
      <td>2</td>
      <td>(6, 6)</td>
      <td>False</td>
      <td>relu</td>
      <td>...</td>
      <td>-0.733490</td>
      <td>-0.736841</td>
      <td>-0.737170</td>
      <td>-0.734712</td>
      <td>-0.737595</td>
      <td>-0.733860</td>
      <td>-0.734644</td>
      <td>-0.735834</td>
      <td>0.001661</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51.766060</td>
      <td>16.999921</td>
      <td>0.070073</td>
      <td>0.028145</td>
      <td>True</td>
      <td>1000</td>
      <td>2</td>
      <td>(9, 9, 9)</td>
      <td>False</td>
      <td>relu</td>
      <td>...</td>
      <td>-0.737034</td>
      <td>-0.740796</td>
      <td>-0.736595</td>
      <td>-0.732730</td>
      <td>-0.736996</td>
      <td>-0.733950</td>
      <td>-0.732615</td>
      <td>-0.735872</td>
      <td>0.002380</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28.950217</td>
      <td>6.136632</td>
      <td>0.078734</td>
      <td>0.028465</td>
      <td>True</td>
      <td>1000</td>
      <td>1</td>
      <td>(4, 4, 4)</td>
      <td>False</td>
      <td>relu</td>
      <td>...</td>
      <td>-0.734982</td>
      <td>-0.738786</td>
      <td>-0.734911</td>
      <td>-0.736305</td>
      <td>-0.737246</td>
      <td>-0.734994</td>
      <td>-0.735775</td>
      <td>-0.735925</td>
      <td>0.001385</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 24 columns</p>
</div>""", unsafe_allow_html=True
    )

st.markdown("""<br><hr><br>""", unsafe_allow_html=True)

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

with open('Models/best_estimator_mlp.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

y_pred = model.predict(X_test)
tolerance = 300
correct_predictions = 0
for pred, actual in zip(y_pred, y_test):
    if abs(pred - actual) <= tolerance:
        correct_predictions += 1
accuracy = correct_predictions / len(y_test)
st.success(f"Accuracy: {accuracy * 100:.2f}%")
st.success(f"RMS: {np.sqrt(np.mean(np.square(np.subtract(np.log1p(y_test), np.log1p(y_pred))))) :.3f}")

st.markdown("""<br><hr><br>""", unsafe_allow_html=True)
