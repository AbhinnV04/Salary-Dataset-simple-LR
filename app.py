import streamlit as st 
import matplotlib.pyplot as plt 
import pickle 
import numpy as np 
import pandas as pd


def load_model():
    with open('model.pk1', 'rb') as file:
        data = pickle.load(file)
        
    return data 


data = load_model()
reg = data['model']

def load_data():
    df = pd.read_csv('Salary_dataset.csv')
    df['Salary'] = df['Salary']
    df = df.rename(columns={'YearsExperience':'Experience(Years)', 'Salary':'Salary($10000)'})
    df = df.drop('Unnamed: 0', axis=1)
    
    fig, ax = plt.subplots(1,1)
    ax = plt.scatter(data=df, x='Experience(Years)', y='Salary($10000)',marker='x')
    plt.ylabel('Salary (USD $)')
    plt.xlabel('Experience (years)')
    plt.title('Salary Dataset visaulized')
    st.pyplot(fig)
    

def show_page():
    st.title('Salary Dataset - Simple linear regression')
    st.write("""
             Dataset link - https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression
             ### Let's make a prediction
             """)
    number = st.number_input('Enter work experience (years)')
    number = np.array(number).reshape(1, -1)
    
    calculate = st.button('Calculate')
    if calculate:
        salary = reg.predict(number)
        st.subheader(f"The estimated Salary is ${salary[0][0]*10_000:.2f}")
    
    st.write("# Dataset Visualized")
    load_data()
    
    
show_page()