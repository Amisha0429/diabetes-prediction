# Diabetes Prediction WebApp using Machine learning and Python

# Import libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

# To create a title and a sub-title
st.write("""
# Diabetes Detection
Detect if someone has diabetes using machine learning and python
""")

# To open and display and image on webapp
image = Image.open('/Users/amisha/Desktop/DiabetesPrediction/Doctor.png')
st.image(image, caption='ML', use_column_width = True)

# To get the data
df = pd.read_csv('/Users/amisha/Desktop/DiabetesPrediction/Diabetes.csv')

# To set a sub-header on the web app
st.subheader('Data information')

# To show the data as a table
st.dataframe(df)

# To show some statistics on the data
st.write(df.describe())

# To show the data as a chart
chart = st.bar_chart(df)

# To split the data into independent 'x' and dependent 'y' variables
x = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values

# To split the dataset into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# To get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.5)
    bmi = st.sidebar.slider('bmi', 0.0, 67.1, 30.0)
    diabetes_pedigree_function = st.sidebar.slider('diabetes_pedigree_function', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)
    
    user_data = {'pregnancies':pregnancies,
                 'glucose':glucose,
                 'blood_pressure':blood_pressure,
                 'skin_thickness':skin_thickness,
                 'insulin':insulin,
                 'bmi':bmi,
                 'diabetes_pedigree_function':diabetes_pedigree_function,
                 'age':age
                 }
    
    features = pd.DataFrame(user_data, index = [0])
    return features
 
# To store the user input into a variable
user_input = get_user_input()

# To set a subheader and display the users input
st.subheader('User Input:')
st.write(user_input)

# To craete and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train, y_train)

# To show the models metrics
st.subheader('Model Test Accuracy Score:')
st.write( str(accuracy_score(y_test, RandomForestClassifier.predict(x_test))*100)+'%' )

# To store the model's prediction in a variable
prediction = RandomForestClassifier.predict(user_input)

# To set a sub header and display the classifictaion
st.subheader('Classification')
st.write(prediction)
if prediction == 1:
    st.write('Person has Diabetes.')
else:
    st.write('Person does not have Diabetes')