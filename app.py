import streamlit as st
import pickle
import numpy as np

# Load the trained Random Forest model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Define the input fields based on the dataset columns, arranged in a 6x2 grid
def user_input_features():
    col1, col2 = st.columns(2)

    with col1:
        Time = st.number_input('Time', min_value=0.0, format="%.6f", step=0.000001)
        Ipv = st.number_input('Ipv', format="%.6f", step=0.000001)
        Vdc = st.number_input('Vdc', format="%.6f", step=0.000001)
        va = st.number_input('va', format="%.6f", step=0.000001)
        vc = st.number_input('vc', format="%.6f", step=0.000001)
        If = st.number_input('If', format="%.6f", step=0.000001)

    with col2:
        mode = st.selectbox('Mode', options=[0, 1])
        Vpv = st.number_input('Vpv', format="%.6f", step=0.000001)
        vb = st.number_input('vb', format="%.6f", step=0.000001)
        Iabc = st.number_input('Iabc', format="%.6f", step=0.000001)
        Vabc = st.number_input('Vabc', format="%.6f", step=0.000001)
        Vf = st.number_input('Vf', format="%.6f", step=0.000001)

    # Create a dictionary to store user inputs
    data = {
        'Time': Time,
        'mode': mode,
        'Ipv': Ipv,
        'Vpv': Vpv,
        'Vdc': Vdc,
        'va': va,
        'vb': vb,
        'vc': vc,
        'Iabc': Iabc,
        'If': If,
        'Vabc': Vabc,
        'Vf': Vf
    }

    features = np.array(list(data.values())).reshape(1, -1)
    return features


# Streamlit app title
st.title("Fault Detection using Random Forest")

# Input form to collect feature values from the user
st.header('Input the system values:')
input_features = user_input_features()

# Button for prediction
if st.button('Predict Fault Type'):
    # Perform prediction using the Random Forest model
    prediction = model.predict(input_features)

    # Update fault_map with numerical keys
    fault_map = {
        'F0': 'Fault Type 0',
        'F1': 'Fault Type 1',
        'F2': 'Fault Type 2',
        'F3': 'Fault Type 3',
        'F4': 'Fault Type 4',
        'F5': 'Fault Type 5',
        'F6': 'Fault Type 6',
        'F7': 'Fault Type 7'
    }

    # Display the fault type based on the model's prediction
    st.write(f"The detected fault type is: {fault_map[prediction[0]]}")
