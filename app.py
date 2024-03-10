import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# Function to make predictions
def predict_fare(features):
    df = pd.read_csv('new_uber.csv')
    df1 = pd.read_csv('uber_dataset.csv')
    x = df[['source', 'destination', 'cab_type', 'name']]
    y = df1['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    random = RandomForestRegressor(n_estimators=100, random_state=0)
    random.fit(x_train, y_train)
    # random.score(x_test, y_test)
        # Make predictions
    prediction = random.predict(features)
    return prediction
def Encode_source(Source):
    if Source=='Back Bay':
        source=0
    elif Source=="Beacon Hill":
        source=1
    elif Source=='Boston University':
        source=2
    elif Source=='Fenway':
        source=3
    elif Source=='Financial District':
        source=4
    elif Source=='Haymarket Square':
        source=5
    elif Source=='Theatre District':
        source=10
    elif Source=='West End':
        source=11
    elif Source=='North End':
        source=6
    elif Source=='North Station':
        source=7
    elif Source=='Northeastern University':
        source=8
    elif Source=='South Station ':
        source=9
    else:
        source= 0
    return source
def Encode_destination(Destination):
    if Destination=='Back Bay':
        destination=0
    elif Destination=="Beacon Hill":
        destination=1
    elif Destination=='Boston University':
        destination=2
    elif Destination=='Fenway':
        destination=3
    elif Destination=='Financial District':
        destination=4
    elif Destination=='Haymarket Square':
        destination=5
    elif Destination=='Theatre District':
        destination=10
    elif Destination=='West End':
        destination=11
    elif Destination=='North End':
        destination=6
    elif Destination=='North Station':
        destination=7
    elif Destination=='Northeastern University':
        destination=8
    elif Destination=='South Station ':
        destination=9
    else:
        destination= 0
    return destination
def Encode_cab(Cab_types):
    if Cab_types=='Uber':
        cab_types=1
    elif Cab_types=='Lyft':
        cab_types=0
    else:
        cab_types=0
    return  cab_types
def Encode_name(Name):
    if Name=='Black SUV':
        name=0
    elif Name=="Taxi":
        name=3
    elif Name=='Uber X':
        name=5
    elif Name==   'UberPool':
        name=4
    elif Name=='Lux':
        name=1

    elif Name=='Shared':
        name=2
    else:
        name= 0
    return name
def main():
    # Set title and description
    st.title('Uber Fare Prediction')
    st.write('This app predicts the fare for an Uber ride based on input features in Massachusetts,USA.')
    Source=st.selectbox(
        'Select ur source place',
    ('Select Source','Financial District', 'Theatre District', 'Back Bay','North End','Boston University ','Fenway','Northeastern University',
     'South Station ','Haymarket Square ','West End','Beacon Hill','North Station')
    )
    Destination=st.selectbox('select your destination',
                             ('Select Destination','Financial District', 'Theatre District', 'Back Bay', 'North End', 'Boston University ',
                              'Fenway', 'Northeastern University',
                              'South Station ', 'Haymarket Square ', 'West End', 'Beacon Hill', 'North Station')
                             )
    Cab_types=st.selectbox("Select Cab TYpe",
                           ('Select cab_type','Uber','Lyft'))
    Name=st.selectbox("Cab Name",
                      ('Select Cab_name','Black SUV',"Taxi",'Uber X','UberPool','Lux',"Shared"))
    source=Encode_source(Source)
    destination=Encode_destination(Destination)
    cab_types=Encode_cab(Cab_types)
    name=Encode_name(Name)
    Features=[[source,destination,cab_types,name]]
    # Predict the fare
    if st.button('Predict Fare'):
        prediction = predict_fare(Features)
        st.write(f'Predicted Fare:{round(prediction[0],2)} $')
# Run the app
if __name__ == '__main__':
    main()

