from pandasql import sqldf
from typing import List, Dict
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def pysqldf(q, dataframes):
    return sqldf(q, dataframes)

#df_table = st.session_state.uploaded_data

tokenizer = AutoTokenizer.from_pretrained("juierror/flan-t5-text2sql-with-schema-v2")
model = AutoModelForSeq2SeqLM.from_pretrained("juierror/flan-t5-text2sql-with-schema-v2")

def get_prompt(tables, question):
    prompt = f"""convert question and table into SQL query. tables: {tables}. question: {question}"""
    return prompt

def prepare_input(question: str, tables: Dict[str, List[str]]):
    tables = [f"""{table_name}({",".join(tables[table_name])})""" for table_name in tables]
    tables = ", ".join(tables)
    prompt = get_prompt(tables, question)
    input_ids = tokenizer(prompt, max_length=512, return_tensors="pt").input_ids
    return input_ids

def inference(question: str, tables: Dict[str, List[str]]) -> str:
    input_data = prepare_input(question=question, tables=tables)
    input_data = input_data.to(model.device)
    outputs = model.generate(inputs=input_data, num_beams=10, max_length=512) #top_k=10,
    result = tokenizer.decode(token_ids=outputs[0], skip_special_tokens=True)
    return result

def user_query_dataframe(question: str, df_table: pd.DataFrame) -> (str, pd.DataFrame):
    """
    Receives a string (question) and a dataframe
    Pass the question to the model inference and apply the result query to the given dataframe

    """
    try:
        columns_list = df_table.columns.tolist()
        df_table_schema = {"df_table": columns_list}
        query = inference(question, df_table_schema)
        print(query)
        result_df = pysqldf(query, locals())
        return ('table2',result_df)
    except Exception as e:
        print(f"Error executing SQL query: {str(e)}")
        return ('0', question)  # Return an empty DataFrame or handle the error as needed

"""
print(inference("how many people with name jui and age less than 25", {
    "people_name": ["id", "name"],
    "people_age": ["people_id", "age"]
}))

print(inference("what is id with name jui and age less than 25", {
    "df_table": ["id", "TGCG", "age", "lifestage"]
}))

"""
