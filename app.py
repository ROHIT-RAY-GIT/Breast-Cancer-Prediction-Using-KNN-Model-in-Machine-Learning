import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


# Load the saved KNN model
with open('KNN_7_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

# Title of the application
st.title("Breast Cancer Prediction")
st.markdown("Input the following tumor characteristics and click **Predict** to determine whether the tumor is benign or malignant.")

# Input fields for the features on the main screen
clump_thickness = st.slider('Clump Thickness', 1, 10, 1)
uniformity_cell_size = st.slider('Uniformity of Cell Size', 1, 10, 1)
uniformity_cell_shape = st.slider('Uniformity of Cell Shape', 1, 10, 1)
marginal_adhesion = st.slider('Marginal Adhesion', 1, 10, 1)
single_epithelial_cell_size = st.slider('Single Epithelial Cell Size', 1, 10, 1)
bare_nuclei = st.slider('Bare Nuclei', 1, 10, 1)
bland_chromatin = st.slider('Bland Chromatin', 1, 10, 1)
normal_nucleoli = st.slider('Normal Nucleoli', 1, 10, 1)
mitoses = st.slider('Mitoses', 1, 10, 1)

# Collect the user input into a feature vector
input_data = [[clump_thickness, uniformity_cell_size, uniformity_cell_shape, 
               marginal_adhesion, single_epithelial_cell_size, bare_nuclei, 
               bland_chromatin, normal_nucleoli, mitoses]]

# Feature scaling
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = knn_model.predict(input_data_scaled)

    # Display prediction result
    if prediction == 2:
        st.success("The tumor is predicted to be **Benign**.")
    else:
        st.error("The tumor is predicted to be **Malignant**.")

# Footer
st.markdown("---")
st.write("Created with ❤️ by Rohit")
