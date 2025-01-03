import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import requests

# Function to fetch data from an API
@st.cache_data
def fetch_data_from_api(api_url, headers=None):
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()  # Parse and return JSON response
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        st.error(f"An error occurred: {err}")
        return None

# Load the model
filename = 'final_model.joblib'
try:
    loaded_model = joblib.load(filename)
except FileNotFoundError:
    st.error("Model file not found. Ensure 'final_model.joblib' exists.")
    st.stop()

# Load or fetch the data
api_url = "https://api.example.com/customer-data"  # Replace with your API endpoint
headers = {"Authorization": "Bearer YOUR_API_KEY"}  # Replace with your actual API key
api_data = fetch_data_from_api(api_url, headers)

if api_data:
    df = pd.DataFrame(api_data)
else:
    try:
        df = pd.read_csv("Clustered_Customer_Data.csv")
    except FileNotFoundError:
        st.error("Data file not found. Ensure 'Clustered_Customer_Data.csv' exists.")
        st.stop()

# Set up Streamlit app
st.markdown('<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)
st.title("Customer Cluster Prediction")

# Input form
with st.form("my_form"):
    balance = st.number_input('Balance', step=0.001, format="%.6f")
    balance_frequency = st.number_input('Balance Frequency', step=0.001, format="%.6f")
    purchases = st.number_input('Purchases', step=0.01, format="%.2f")
    oneoff_purchases = st.number_input('OneOff Purchases', step=0.01, format="%.2f")
    installments_purchases = st.number_input('Installments Purchases', step=0.01, format="%.2f")
    cash_advance = st.number_input('Cash Advance', step=0.01, format="%.6f")
    purchases_frequency = st.number_input('Purchases Frequency', step=0.01, format="%.6f")
    oneoff_purchases_frequency = st.number_input('OneOff Purchases Frequency', step=0.1, format="%.6f")
    purchases_installment_frequency = st.number_input('Purchases Installments Frequency', step=0.1, format="%.6f")
    cash_advance_frequency = st.number_input('Cash Advance Frequency', step=0.1, format="%.6f")
    cash_advance_trx = st.number_input('Cash Advance Transactions', step=1)
    purchases_trx = st.number_input('Purchases Transactions', step=1)
    credit_limit = st.number_input('Credit Limit', step=0.1, format="%.1f")
    payments = st.number_input('Payments', step=0.01, format="%.6f")
    minimum_payments = st.number_input('Minimum Payments', step=0.01, format="%.6f")
    prc_full_payment = st.number_input('PRC Full Payment', step=0.01, format="%.6f")
    tenure = st.number_input('Tenure', step=1)

    data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, 
             cash_advance, purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency, 
             cash_advance_frequency, cash_advance_trx, purchases_trx, credit_limit, payments, 
             minimum_payments, prc_full_payment, tenure]]

    submitted = st.form_submit_button("Submit")

# Prediction and Visualization
if submitted:
    try:
        cluster = loaded_model.predict(data)[0]
        st.success(f"Data belongs to Cluster {cluster}")

        # Filter data for the cluster
        cluster_df = df[df['Cluster'] == cluster]

        # Visualize features in the cluster
        st.write(f"Cluster {cluster} Data Overview:")
        st.write(cluster_df.describe())

        for feature in cluster_df.drop(['Cluster'], axis=1).columns:
            plt.figure(figsize=(5, 3))
            sns.histplot(cluster_df[feature], kde=True, bins=20)
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            st.pyplot(plt)
            plt.clf()

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
