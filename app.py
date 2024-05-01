import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = load_model('./my_model.keras')

# Define the feature columns
FEATURES = ['store_nbr', 'family', 'month']

# Load the LabelEncoder for 'family' feature
label_encoder = LabelEncoder()
label_encoder.classes_ = pd.read_csv('./train.csv')['family'].unique()

# Create the Streamlit app
st.title("Sales Prediction App")

# Get user inputs
store_nbr = st.number_input("Store Number", min_value=1, step=1)
family = st.selectbox("Family", label_encoder.classes_)
month = st.number_input("Month", min_value=1, max_value=12, step=1)

# Create a dictionary with the user inputs
user_input = {
    'store_nbr': [store_nbr],
    'family': [label_encoder.transform([family])[0]],
    'month': [month],
}

# Create a DataFrame from the user input
X_user = pd.DataFrame(user_input)

# Predict the sales
if st.button("Predict Sales"):
    prediction = model.predict(X_user.astype('float32'))[0][0]
    st.write(f"Predicted Sales: {prediction:.2f}")