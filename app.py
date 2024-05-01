import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('./my_model.keras')

# Load the LabelEncoder for 'family' feature
label_encoder = LabelEncoder()
label_encoder.classes_ = pd.read_csv('./train.csv')['family'].unique()

# Define the feature columns
FEATURES = ['store_nbr', 'family', 'month']

# Define app title and favicon
st.set_page_config(page_title="Sales Prediction Model", page_icon=":bar_chart:")

# Define app title and subtitle
st.title("Sales Prediction")
st.markdown("""
Predict the future sales of a store
""")


# Get user inputs
st.header("User Inputs")
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
    st.success(f"Predicted Sales: {prediction:.2f}")