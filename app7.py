import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pickle
model=pickle.load(open('Logistic_Regression_model.pkl','rb'))
st.title('Titanic Survival Prediction')

# collect user input
pclass = st.selectbox('Passenger Class (Pclass)',[1,2,3])
sex=st.selectbox('Sex',['male','Female'])
age=st.slider('Age',0,80,30)
fare=st.slider('Fare',0.0,500.0,50.0)
sibsp=st.number_input('Siblings/Spouses Aboard',0,10,0)
parch=st.number_input('Parents/Children Aboard',0,10,0)
embarked=st.selectbox('Port of Embarkation',['S','C','Q'])

# Encode inputs
sex=0 if sex == 'male' else 1
embarked_C= 1 if embarked == 'C' else 0
embarked_Q= 1 if embarked == 'Q' else 0

# Prepare input array
features=np.array([[pclass,sex,age,sibsp,parch,fare,embarked_C,embarked_Q]])

# Make prediction
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0][1]

# Display result
st.subheader('Prediction Result')
st.write('Survived' if prediction == 1 else 'Did Not Survive' )
st.write(f'Survival Probability : {probability:.2f}')
