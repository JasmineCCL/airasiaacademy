import streamlit as st
import pandas as pd
import pickle

st.write("""
# Advertising Sales Prediction

This app predicts the **Advertising** sales!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('Number of TV', 0, 100, 100)
    Radio = st.sidebar.slider('Number of Radio',0, 300, 10)
    Newspaper = st.sidebar.slider('Number of Newspaper', 0, 200, 50)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("sales.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(f"{prediction[0]:.2f}")
