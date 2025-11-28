import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # Predicción de vida fitness ''')
st.image("ejemplo_a_seguir.jpg", caption="Introduce tus datos para saber si llevas una vida fitness, o no")

st.header('Datos de evaluación')

def user_input_features():
  # Entrada
  age = st.number_input('age:', min_value=1, max_value=100, value = 1, step = 1)
  height_cm = st.number_input('height_cm:', min_value=1, max_value=240, value = 1, step = 1)
  weight_kg = st.number_input('weight_kg:', min_value=1, max_value=200, value = 1, step = 1)
  Heart_rate = st.number_input('Heart_rate:',min_value=1, max_value=160, value = 1, step = 1)
  blood_pressure = st.number_input('blood_pressure:', min_value=1, max_value=150, value = 1, step = 1)
  nutrition_quality = st.number_input('nutrition_quality:', min_value=1, max_value=10, value = 1, step = 1)
  activity_index = st.number_input('activity_index:', min_value=1, max_value=4, value = 1, step = 1)
  gender = st.number_input('gender:', min_value=0, max_value=1, value = 0, step = 1)

  user_input_data = {'age': age,
                     'gender': gender,
                     'height_cm': height_cm,
                     'weight_kg': weight_kg,
                     'Heart_rate': Heart_rate,
                     'blood_pressure': blood_pressure,
                     'nutrition_quality': nutrition_quality,
                     'activity_index': activity_index,}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

is_fit =  pd.read_csv('Fitness_Classification2.csv', encoding='latin-1')
X = is_fit.drop(columns='is_fit')
Y = is_fit['is_fit']

classifier = DecisionTreeClassifier(max_depth=6, criterion='entropy', min_samples_leaf=25, max_features=7, random_state=0)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('no es fitness')
elif prediction == 1:
  st.write('si es fitness')
else:
  st.write('Sin predicción')
