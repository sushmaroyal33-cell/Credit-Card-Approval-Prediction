import streamlit as st

st.set_page_config(page_title="Credit Card Approval Prediction")

st.title("Credit Card Approval Prediction System")

age = st.number_input("Age", 18, 100, 25)
income = st.number_input("Annual Income", 10000, 1000000, 50000)
credit_score = st.number_input("Credit Score", 300, 900, 700)
loan_amount = st.number_input("Loan Amount", 1000, 500000, 50000)

if st.button("Predict"):
    if credit_score >= 650 and income >= 30000:
        st.success("✅ Credit Card Approved")
    else:
        st.error("❌ Credit Card Rejected")