import streamlit as st
import languagemodels as lm
import re
from app_functions import *
from pandas_sql_query import *

st.set_page_config(page_title="Chatbot", layout="wide", page_icon= "🕷️")


st.title("Chatbot")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# CSS to change background color
st.markdown(
    """
    <style>
    .rules-box-top {
        background-color: #e6ffe6;  # light green
        border-radius: 5px;
        padding: 10px;
    }
    .rules-box-bottom {
        background-color: #ffe6e6;  # light red
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#st.write(st.session_state.corpus)

#st.write(st.session_state.reference_dict)

try:
    corpus = st.session_state.corpus

    reference_dict= st.session_state.reference_dict

    df_table = st.session_state.uploaded_data

    columns_list = df_table.columns.tolist()

    df_table_schema = {"df_table": columns_list}

    lm.set_max_ram('4gb')
    #lm.store_doc(lm.get_wiki("Planet Saturn"))
    #lm.store_doc('The KPI acceptors has an uplift_value of *xgfw|a3|23')
    #lm.store_doc('The KPI vgtr has an uplift of 45')
    lm.store_doc(corpus)
    #st.write(lm.models.get_model_name('instruct'))

    def check_string(s): #check if a string contains the special characters of the keys defined
        if re.search(r'\*xgfw[^|]*\|[^|]*\|', s):
            return True
        else:
            return False

    def assist(question):
        
        context = lm.get_doc_context(question)#.replace(": ", " - ")

        response = lm.do(f"Answer using context: {context} Question: {question}")
        if check_string(response):
            return response
        else:
            output_sql = user_query_dataframe(question, df_table)
            if output_sql[0] == 'table2':
                return output_sql
            else:
                return response

    def process_response(response):
        if not isinstance(response, str):
            return response
        elif check_string(response):
            key_d = re.search(r'\*xgfw\|([^|]*)\|', response).group(1) 
            return reference_dict[key_d] 
        else: #instead of response we can do the sql process in this step
            return ('0', response)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    #the content is different for the assistant because we need to know the data type of the answer to show it correctly
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["content"][0] == '0' and message["role"] == 'assistant':
                st.markdown(message["content"][1]) 
            elif message["content"][0] == 'fig' and message["role"] == 'assistant':
                st.plotly_chart(message["content"][1])
            elif (message["content"][0] == 'table2' or message["content"][0] == 'list') and message["role"] == 'assistant':
                st.write(message["content"][1])
            elif (message["content"][0] == 'table') and message["role"] == 'assistant':
                st.dataframe(message["content"][1].style.apply(highlight_pvalue, axis=1))
            elif message["content"][0] == 'box_top' and message["role"] == 'assistant':
                for rule in message["content"][1][0]:
                    st.markdown(f"<div class='rules-box-top'>{rule}</div>", unsafe_allow_html=True)
                st.dataframe(message["content"][1][1].style.apply(highlight_pvalue, axis=1)) 
            elif message["content"][0] == 'box_bottom' and message["role"] == 'assistant':
                for rule in message["content"][1][0]:
                    st.markdown(f"<div class='rules-box-bottom'>{rule}</div>", unsafe_allow_html=True)
                st.dataframe(message["content"][1][1].style.apply(highlight_pvalue, axis=1)) 
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            #st.write(prompt)
            message_placeholder = st.empty()
            full_response = ""
            #st.write(st.session_state.messages)
            response = assist(prompt)
            if process_response(response)[0] == '0':
                st.write(process_response(response)[1]) 
            elif process_response(response)[0] == 'fig':
                st.plotly_chart(process_response(response)[1])
            elif process_response(response)[0] == 'table2' or  process_response(response)[0] == 'list':
                st.write(process_response(response)[1])
            elif process_response(response)[0] == 'table':
                st.dataframe(process_response(response)[1].style.apply(highlight_pvalue, axis=1))
            elif process_response(response)[0] == 'box_top':
                for rules in process_response(response)[1][0]:
                    st.markdown(f"<div class='rules-box-top'>{rules}</div>", unsafe_allow_html=True)
                st.dataframe(process_response(response)[1][1].style.apply(highlight_pvalue, axis=1))
            elif process_response(response)[0] == 'box_bottom':
                for rules in process_response(response)[1][0]:
                    st.markdown(f"<div class='rules-box-bottom'>{rules}</div>", unsafe_allow_html=True)
                st.dataframe(process_response(response)[1][1].style.apply(highlight_pvalue, axis=1))
            if not isinstance(response, str):
                print(response[1])
                #full_response += response[1]
            else:
                print(response)
                full_response += response

            st.session_state.messages.append({"role": "assistant", "content": process_response(response)})
            #message_placeholder.markdown(full_response + "▌")
            #message_placeholder.markdown(full_response)
        #st.session_state.messages.append({"role": "assistant", "content": full_response})
    
except AttributeError:
    st.write("Dataset not loaded correctly")
    st.write("Click on the Process Data button after uploading the file")