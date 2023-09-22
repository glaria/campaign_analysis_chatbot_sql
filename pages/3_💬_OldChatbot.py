"""
Steps: 
(there is a strange behavior of the language model where it modifies the last value of a key 
    if the exact reference the reference (kpi for example) is not found. 
    To deal with this we can keep something like *xgfw|a3|23, 
    where *xgfw is the special identifier, a3 value identifier, and 23 and some extra digits that can be ignored)

1. After receiving the inputs from the file calculate all the relevant information
2. Store this info somewhere in order to make it available in other pages, maybe using st.session_state
3. Some kind of key should be used when storing this info, something like a3|23.
   We need to use json or dictionary in order to be able to have keys and values. 
   Value can be any type of Python objects, so a reference to the object type needs to be stored as well
4. In parallel to the storing of the info we need to generate the document with the references.
   This document will consist on a set of phrases of the form: 
   "The [element_type] [element_name] has a [category] of [category_value]
   where element_type can be KPI, continuous variable, discrete variable, element_name is its name
   category is a possible type of value like uplift_value, conversion_value... 
   and category_value is special element of the form *xgfwñ|[key] where key is a key from the previous json/dict
5. Whenever we have *xgfw (and certain number of tabs) we need to extract the real value from the table.
   Therefore, we need 2 special functions to deal with the output from response = assist(st.session_state.dialog)
"""

import streamlit as st
import languagemodels as lm
import re

def check_string(s): #check if a string contains the special characters of the keys defined
    if re.search(r'\*xgfw[^|]*\|[^|]*\|', s):
        return True
    else:
        return False


st.write(st.session_state.corpus)

st.write(st.session_state.reference_dict)


corpus = st.session_state.corpus

reference_dict= st.session_state.reference_dict

lm.set_max_ram('4gb')
#lm.store_doc(lm.get_wiki("Planet Saturn"))
#lm.store_doc('The KPI acceptors has an uplift_value of *xgfw|a3|23')
#lm.store_doc('The KPI vgtr has an uplift of 45')
lm.store_doc(corpus)


def assist(question):
    context = lm.get_doc_context(question).replace(": ", " - ")

    return lm.do(f"Answer using context: {context} Question: {question}")

st.title("Chatbot")

def process_response(response):
    if check_string(response):
        key_d = re.search(r'\*xgfw\|([^|]*)\|', response).group(1) 
        return reference_dict[key_d] 
    else:
        return ('0', response)

def reset():
    st.session_state.dialog = ""
    st.session_state.message = ""
    st.session_state.objects = None


# Initialize empty dialog context on first run
if "dialog" not in st.session_state:
    reset()

if st.session_state.message:
    # Add new message to dialog
    st.session_state.dialog += f"User: {st.session_state.message}\n\nAssistant: "
    st.session_state.message = ""

    # Prompt LLM to get response
    #response = lm.chat(f"{st.session_state.dialog}")

    #to simulate a chat we can maybe use store_doc to save previous questions/answers as docs
    response = assist(st.session_state.dialog)

    #after getting the response from the chatbot we need to process it
    #if it contains *xgfw and 2 pipes (|) then we extract the data from the reference dict
    #if not, we return the response value (this behavior can be modified later)

    # Display full dialog
    print(response)
    if process_response(response)[0] == '0':
        st.session_state.dialog += process_response(response)[1] + "\n\n"
    elif process_response(response)[0] == 'fig':
        st.plotly_chart(process_response(response)[1])
    elif process_response(response)[0] == 'table' or  process_response(response)[0] == 'list':
        st.write(process_response(response)[1])

    st.write(st.session_state.dialog)

st.text_input("Message", key="message")

st.button("Reset", on_click=reset)