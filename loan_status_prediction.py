import numpy as np
from PIL import Image
import streamlit as st
import pickle

# Load the model
@st.cache_resource
def load_model():
    with open('loan_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Predict car price
def predict_loan_status(input_data):
    model = load_model()
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction

# Main function
def main():
    # Set page configuration
    st.set_page_config(page_title="Loan Status Prediction", page_icon="💸", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stNumberInput>div>div>input {
                font-size: 16px;
            }
            .stSelectbox>div>div>select {
                font-size: 16px;
            }
            .stMarkdown {
                font-size: 18px;
            }
            .created-by {
                font-size: 20px;
                font-weight: bold;
                color: #4CAF50;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("💸 Loan Status Prediction")
    st.markdown("This app predicts if an individual will be granted a loan based on their information.")

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="created-by">Created by Andrew O.A.</p>', unsafe_allow_html=True)
        
        # Load and display profile picture
        try:
            profile_pic = Image.open("prof.jpeg")  # Replace with your image file path
            st.image(profile_pic, caption="Andrew O.A.", use_container_width=True, output_format="JPEG")
        except:
            st.warning("Profile image not found.")

        st.title("About")
        st.info("This app uses a machine learning model to predict if an individual will be granted loan based on their information.")
        st.markdown("[GitHub](https://github.com/Andrew-oduola) | [LinkedIn](https://linkedin.com/in/andrew-oduola-django-developer)")

    result_placeholder = st.empty()

    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], help="Select the gender")
        married = st.selectbox("Married", ["Yes", "No"], help="Select the marital status")
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], help="Select the number of dependents")
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], help="Select the education level")
        self_employed = st.selectbox("Self Employed", ["Yes", "No"], help="Select if the individual is self-employed")

        
    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=320000, 
                                           help="Enter the applicant's income")
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0, 
                                             help="Enter the coapplicant's income")
        loan_amount = st.number_input("Loan Amount", min_value=0, value=2500000, 
                                      help="Enter the loan amount")
        loan_amount_term = st.number_input("Loan Amount Term", min_value=0, value=180, 
                                           help="Enter the loan amount term")
        credit_history = st.selectbox("Credit History", ["0", "1"], help="Select the credit history")
        
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], help="Select the property area")

    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_area = 1 if property_area == "Urban" else (2 if property_area == "Semiurban" else 0)
    dependents = 4 if dependents == "3+" else int(dependents)
    credit_history = int(credit_history)

    applicant_income = float(applicant_income)/1000
    coapplicant_income = float(coapplicant_income)/1000
    loan_amount = float(loan_amount)/1000

    # Prepare input data for the model
    input_data = [gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]
    input_data = [float(x) for x in input_data]  # Ensure all data is float

    # Prediction button
    if st.button("Predict"):
        try:
            prediction = predict_loan_status(input_data)
            
            if prediction == 1:
                prediction_text = "Loan granted"
                result_placeholder.success(prediction_text)
                st.success(prediction_text)
            else:
                prediction_text = "Loan not granted"
                result_placeholder.error(prediction_text)
                st.error(prediction_text)

            # st.markdown("**Note:** This is a simplified model and may not be accurate for all cases.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            result_placeholder.error("An error occurred during prediction. Please check the input data.")

if __name__ == "__main__":
    main()