import streamlit as st
import time

st.title(":red[Welcome To The Digital Calculator]🧮")

num1 = st.number_input("Enter your first number : ")
num2 = st.number_input("Enter your second number : ")
#ope = st.text_input("Select the operator - add, sub, mul, div, pow: ").lower()
ope = st.selectbox("Choose the operator which you want to perform",("add","sub","mul","div","pow"))

button = st.button("Click Here For The Answer",type="primary")

if button:
    if ope == "add":
        st.write(" 🎉 addition = ",  num1+num2)
        st.balloons()
    elif ope == "sub":
        st.write(" 🎉 subtraction = ",  num1-num2)
        st.snow()
    elif ope == "mul":
        st.write("🎉 multiple = ",  num1*num2)
        st.success("This is a Success Message!", icon="✅")
    elif ope == "div":
        if num2 == 0:
            st.error("Number cannot be divisable by zero so please enter proper number",icon="❌")
        else:
            with st.spinner("Wait for it....⏳",show_time=True):
                time.sleep(2)
                st.write("🎉 division = ",  num1/num2)
    elif ope == "pow":
        with st.status("Calculating power..."):
            st.write("🎉 power = ",  num1**num2)
    else:
        st.warning("This is a Warning",icon="⚠️")
        st.write("Select the operator from the given list: [add, sub, mul, div, pow] ")