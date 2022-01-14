import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import streamlit as st

df = pd.read_csv("train.csv")

column_names = ['Battery_Power', 'Has_Bluetooth', 'Clock_Speed', 'Has_Dual_Sim', 'Front_Camera_Mega_Pixels', 'Has_4G', 
                'Internal_Memory', 'Mobile_Depth', 'Mobile_Weight', 'No_Cores', 'Primary_Camera_Mega_Pixels', 'Pixel_Height', 
                'Pixel_Width', 'RAM', 'Screen_Height', 'Screen_Width', 'Battery_Talk_Time', 'Has_3G', 'Has_Touch_Screen', 
                'Has_WiFi', 'Price_Range']

df.columns = column_names

con_int = ['Battery_Power', 'Has_Bluetooth', 'Has_Dual_Sim', 'Front_Camera_Mega_Pixels', 'Has_4G', 
                'Internal_Memory', 'Mobile_Weight', 'No_Cores', 'Primary_Camera_Mega_Pixels', 'Pixel_Height', 
                'Pixel_Width', 'RAM', 'Screen_Height', 'Screen_Width', 'Battery_Talk_Time', 'Has_3G', 'Has_Touch_Screen', 
                'Has_WiFi', 'Price_Range']

for i in con_int:
    df[i] = df[i].astype(int)

# Splitting the data into Features and Target variables:

X=df.iloc[:,:-1]
y=df.iloc[:,-1:]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=150)

dt = DecisionTreeClassifier()

model = dt.fit(X_train,y_train)

y_pred = model.predict(X_test)

#print(model.predict([[1540,0,0.5,1,18,1,25,0.5,96,8,20,295,1752,3893,10,0,7,1,1,0]]))

#print(classification_report(y_test,y_pred))

st.title("Welcome to the Mobile Price Predictor Web Application.")


st.sidebar.title("Navigation Panel:")
nav = st.sidebar.radio("",["Home","Prediction"])

image_me = "Kalpesh Shinde Profile Picture.png"    

st.sidebar.image(image_me,width=250,caption="Kalpesh Shinde")

#new_title = '<p style="font-family:sans-serif; color:black; font-size: 17px;">KALPESH SHINDE</p>'
#st.sidebar.image(image_me, channels="BGR")
#st.sidebar.markdown(new_title, unsafe_allow_html=True)



st.sidebar.title("Contact:")
st.sidebar.markdown('[![Kalpesh-Shinde]'
                    '(https://img.shields.io/badge/LinkedIn-Kalpesh%20Shinde-orange)]'
                    '(https://www.linkedin.com/in/kalpeshshinde/)')
st.sidebar.markdown('[![Kalpesh-Shinde]'
                    '(https://img.shields.io/badge/GitHub%20-Kalpesh%20Shinde-white)]'
                    '(https://www.github.com/shindekalpesh/)')
st.sidebar.markdown('[![Kalpesh-Shinde]'
                    '(https://img.shields.io/badge/Email-Kalpesh%20Shinde-green)]'
                    '(mailto:kalpeshtheofficial@gmail.com)')


if nav == 'Home':
    st.header("This application is made with Streamlit and love.\nBy **Kalpesh Shinde**")
    st.write("Use this app for predicting the price of a mobile phone with desired features.")
    st.write("Please, proceed to the sidebar of this app to go the prediction page.")
    st.image("image.jpg",width=600)
    

    #link = '[GitHub](http://github.com/shindekalpesh)'
    #st.markdown(link, unsafe_allow_html=True)

    

    




elif nav == 'Prediction':
    st.header("Enter the following details: ")

    battery = st.selectbox("Battery Power: ",[i for i in range(500,5001)])
    bluetooth = st.radio("It has Bluetooth? 'No': 0, 'Yes': 1",[0,1])
    clock_speed = st.slider("Clock speed of the proccessor: ",min_value=0.5,max_value=4.5,step=0.1)
    dual_sim = st.radio("It has Dual Sim Card? 'No': 0, 'Yes': 1",[0,1])
    fc_mp = st.slider("Front Camera Mega Pixel: ",min_value=0,max_value=20,step=1)
    has_4g = st.radio("It has 4G? 'No': 0, 'Yes': 1",[0,1])
    internal_memory = st.selectbox("Enter the internal memory phone has: ",[2**i for i in range(0,8)])
    mobile_depth = st.slider("Select Depth of the Phone: ",min_value=0.1,max_value=1.0,step=0.01)
    mobile_weight = st.slider("Select Weight of the Phone: ",min_value=80,max_value=320,step=10)
    no_cores = st.slider("Number of cores of the processor: ",min_value=1,max_value=16,step=1)
    pc_mp = st.slider("Primary Camera Mega Pixel: ",min_value=0,max_value=72,step=1)
    px_height = st.selectbox("Height of Pixel: ",[i for i in range(500,2401)])
    px_width = st.selectbox("Width of Pixel: ",[i for i in range(200,1801)])
    ram = st.selectbox("Height of Screen: ",[i for i in range(256,8001)])
    sc_height = st.selectbox("Height of Screen: ",[i for i in range(5,21)])
    sc_width = st.selectbox("Width of Screen: ",[i for i in range(0,19)])
    battery_talk_time = st.selectbox("Talk Time of Battery: ",[i for i in range(0,50)])
    has_3g = st.radio("It has 3G? 'No': 0, 'Yes': 1",[0,1])
    has_touchscreen = st.radio("It has Touch Screen? 'No': 0, 'Yes': 1",[0,1])
    has_wifi = st.radio("It has WiFi? 'No': 0, 'Yes': 1",[0,1])

    if st.button("Predict Price"):
        result = model.predict([[battery,bluetooth,clock_speed,dual_sim,fc_mp,has_4g,internal_memory,mobile_depth,mobile_weight,no_cores,pc_mp,px_height,px_width,ram,sc_height,sc_width,battery_talk_time,has_3g,has_touchscreen,has_wifi]])
        if result == 0:
            st.success("The price range your phone falls under is **'Entry level category'**.")
        elif result == 1:
            st.success("The price range your phone falls under is **'Mid level category'**.")
        elif result == 2:
            st.success("The price range your phone falls under is **'Pro level category'**.")
        elif result == 3:
            st.success("The price range your phone falls under is **'Flagship level category'**.")



        #st.success(result)
        #st.success("The Predicted Price of the Phone is {result}")
        st.success(f"The accuracy of the model is {accuracy_score(y_test,y_pred)*100} %.")