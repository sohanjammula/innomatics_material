import streamlit as st

# What is a session in streamlit?
## Creating a streamlit app and open it in a browser is referred as SESSION.
## Each time opening a newtab in the browser and go to streamlit app, a new session is created. These sessions are independent.
## A session is a python object that exists in memory to used between runs. 
## Session object exist in memory as long as the user keeps the browser tab open and the connection between front-end and back-end is active.

# What is a session state?
## A way to share variables between runs. Streamlit manipulates the session state through callbacks

st.title("Session State")

count = 0
st.write('Counter', count + 1)

st.write('Session state object')
st.write(st.session_state)

if 'count_state' not in st.session_state:
    st.session_state['count_state'] = 0
else:
    st.session_state['count_state'] += 1
    st.write('Counter_State', st.session_state['count_state'])

button = st.button('Update state')
if 'clicks' not in st.session_state:
    st.session_state['clicks'] = 0

if button:
    st.session_state['clicks'] += 1
    st.write('Counter and Button States', st.session_state['count_state'], st.session_state['clicks'])

number = st.slider('Value', 1, 10, key = 'number')

st.write(f'Counter: {st.session_state['count_state']} Button: {st.session_state['clicks']} and Number: {st.session_state['number']}')

st.write('Session state object')
st.write(st.session_state)

# Wht is a callback?
## A callback is a function that gets called when an user interacts with widget
## We use callbacks through on_change or on_click parameters of the widget
## on_change and on_click parameters accept a function name as an argument. This function is called callback.

st.subheader("Distance Conversion")

def miles_to_km():
    st.session_state['km'] = st.session_state['miles']*1.609

def km_to_miles():
    st.session_state['miles'] = st.session_state['km']*0.621

col1, buffer, col2 = st.columns([2, 1, 2])

with col1:
    miles = st.number_input('Miles', key = 'miles', on_change = miles_to_km)

with col2:
    km = st.number_input('Kilometers', key = 'km', on_change = km_to_miles)

