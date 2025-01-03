import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model using joblib
filename = 'final_model.joblib'  # Make sure your model is saved as .joblib
loaded_model = joblib.load(filename)

# Read the dataset (ensure this file exists)
df = pd.read_csv("Clustered_Customer_Data.csv")

# Set Streamlit options and styling
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown('<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)
st.title("Prediction")

# Form for input fields
with st.form("my_form"):
    balance = st.number_input(label='Balance', step=0.001, format="%.6f")
    balance_frequency = st.number_input(label='Balance Frequency', step=0.001, format="%.6f")
    purchases = st.number_input(label='Purchases', step=0.01, format="%.2f")
    oneoff_purchases = st.number_input(label='OneOff Purchases', step=0.01, format="%.2f")
    installments_purchases = st.number_input(label='Installments Purchases', step=0.01, format="%.2f")
    cash_advance = st.number_input(label='Cash Advance', step=0.01, format="%.6f")
    purchases_frequency = st.number_input(label='Purchases Frequency', step=0.01, format="%.6f")
    oneoff_purchases_frequency = st.number_input(label='OneOff Purchases Frequency', step=0.1, format="%.6f")
    purchases_installment_frequency = st.number_input(label='Purchases Installments Frequency', step=0.1, format="%.6f")
    cash_advance_frequency = st.number_input(label='Cash Advance Frequency', step=0.1, format="%.6f")
    cash_advance_trx = st.number_input(label='Cash Advance Trx', step=1)
    purchases_trx = st.number_input(label='Purchases TRX', step=1)
    credit_limit = st.number_input(label='Credit Limit', step=0.1, format="%.1f")
    payments = st.number_input(label='Payments', step=0.01, format="%.6f")
    minimum_payments = st.number_input(label='Minimum Payments', step=0.01, format="%.6f")
    prc_full_payment = st.number_input(label='PRC Full Payment', step=0.01, format="%.6f")
    tenure = st.number_input(label='Tenure', step=1)

    # Collecting user inputs into a list
    data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance,
             purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency, cash_advance_frequency,
             cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure]]

    # Form submission button
    submitted = st.form_submit_button("Submit")

if submitted:
    # Predict the cluster for the input data
    clust = loaded_model.predict(data)[0]
    st.write(f'Your data belongs to Cluster {clust}')

    # Filter the dataframe for the selected cluster
    cluster_df1 = df[df['Cluster'] == clust]

    # Plot histograms for features in the selected cluster
    plt.rcParams["figure.figsize"] = (20, 3)
    for c in cluster_df1.drop(['Cluster'], axis=1):
        fig, ax = plt.subplots()
        grid = sns.FacetGrid(cluster_df1, col='Cluster')
        grid = grid.map(plt.hist, c)
        plt.show()
        st.pyplot(figsize=(5, 5))
