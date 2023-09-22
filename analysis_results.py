import streamlit as st
import plotly.express as px
from app_functions import *
import itertools
import plotly.graph_objects as go

"""
here we define the analysis results methods that will be used as input in the chatbot
"""

corpus = "" #corpus doc for the chatbot
reference_dict = {} #reference dict to extract python objects

# dataset and information_dataset
#dataset = pd.read_csv("pages/temp/uploaded_data.csv", sep = ',')  
#information_dataset = pd.read_csv("pages/temp/user_defined_info_dataset.csv", sep = ',') 
dataset = st.session_state.uploaded_data
information_dataset = st.session_state.user_defined_info_dataset

# Extract the TGCG column from the dataset
tgcg_column = information_dataset.loc[information_dataset['METATYPE'] == 'TGCG', 'COLUMN'].values[0]

# Set to lower TGCG column: TARGET-> target, Control -> control
dataset[tgcg_column] = dataset[tgcg_column].str.lower()

tgcg_counts = dataset[tgcg_column].value_counts()

# Create a pie chart with counts and percentages
fig = px.pie(tgcg_counts.reset_index(), values=tgcg_counts, names = tgcg_column, title="Target vs Control Groups")

# Display the total counts and percentages in the labels
fig.update_traces(textinfo='label+percent', textfont_size=12, insidetextorientation='radial')

reference_dict['a111'] = ('fig',fig) #st.plotly_chart(fig)

corpus += f"""\n Question: What is the Target/Control distribution? 
              Answer: The Target/Control distribution is *xgfw|a111|1111.

              Question: What is the Target/Control split? 
              Answer: The Target/Control split is *xgfw|a111|1111."""
# Get all KPI columns
kpi_columns = information_dataset.loc[information_dataset['METATYPE'] == 'KPI', 'COLUMN'].values
kpi_columns = kpi_columns.tolist()
reference_dict['a112'] = ('list',kpi_columns) #st.plotly_chart(fig)

corpus += f"""\nQuestion: What is the list of KPIs?
Answer: The list of KPIs is *xgfw|a112|1112.

Question: Can you provide the details of the KPIs list?
Answer: The details for the KPIs list are *xgfw|a112|1112.

Question: How is the list of KPIs defined?
Answer: The list of KPIs is defined as *xgfw|a112|1112."""


# Calculate metrics for each KPI and store the results in a list
results = calculate_metrics(dataset, kpi_columns, tgcg_column) 

# Convert the list of results to a pandas DataFrame
result_df = pd.DataFrame(results, columns=["KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"])
result_df = result_df.map(format_float)

kpi = None
for kpi in kpi_columns:
    key_d = '*xgfw|k' + str(111 + kpi_columns.index(kpi)) + '|' + str(1111 + kpi_columns.index(kpi))
    key_d2 = 'k' + str(111 + kpi_columns.index(kpi))
    reference_dict[key_d2] = ('table',result_df[result_df['KPI'] == kpi])
    corpus += f"""\nQuestion: What is the uplift of the KPI {kpi}?
            Answer: The uplift of the KPI {kpi} is {key_d}.

            Question: Can you provide the uplift value for the KPI {kpi}?
            Answer: The uplift value for the KPI {kpi} is {key_d}.

            Question: How is the uplift of the KPI {kpi} defined?
            Answer: The uplift of the KPI {kpi} is defined as {key_d}."""

    corpus += f"""\nQuestion: What are the results of the KPI {kpi}?
            Answer: The results of the KPI {kpi} are {key_d}.
            Question: Can you provide the results for the KPI {kpi}?

            Answer: The results for the KPI {kpi} are {key_d}.
            Question: How are the results of the KPI {kpi} defined?
            
            Answer: The results of the KPI {kpi} are defined as {key_d}."""


if kpi is not None:
    key_d = '*xgfw|k' + str(111 + kpi_columns.index(kpi)+1) + '|' + str(1111 + kpi_columns.index(kpi)+1)
    key_d2 = 'k' + str(111 + kpi_columns.index(kpi)+1)
    corpus += f"""\nQuestion: What is the uplift of all the KPIs?
    Answer: The uplift of all the KPIs is {key_d}.\n"""

    corpus += f"""\nQuestion: Can you provide the uplift value for all the KPIs?
    Answer: The uplift value for all the KPIs is {key_d}.\n"""

    corpus += f"""\nQuestion: How is the uplift of all the KPIs defined?
    Answer: The uplift of all the KPIs is defined as {key_d}.\n"""

    corpus += f"""\nQuestion: What are the results of all the KPIs?
    Answer: The results of all the KPIs are {key_d}.\n"""

    corpus += f"""\nQuestion: Can you provide the results for all the KPIs?
    Answer: The results for all the KPIs are {key_d}.\n"""

    corpus += f"""\nQuestion: How are the results of all the KPIs defined?
    Answer: The results of all the KPIs are defined as {key_d}.\n"""

    reference_dict[key_d2] = ('table',result_df)



# Segment fields
#st.markdown(f"# Discrete Segments with significant results")
# 1. Identify the segmentation columns

segmentation_columns = information_dataset.loc[(information_dataset['METATYPE'] == 'SF') & 
                                              (information_dataset['DATATYPE'].isin(['NUM_ST', 'BOOL', 'STRING'])), 
                                              'COLUMN'].values
continue_segmentation_columns = information_dataset.loc[(information_dataset['METATYPE'] == 'SF') & 
                                              (information_dataset['DATATYPE'].isin(['NUMERIC'])), 
                                              'COLUMN'].values

segmentation_df = pd.DataFrame({'Field': segmentation_columns})
segmentation_df['Data type'] = 'Discrete'

continue_segmentation_df = pd.DataFrame({'Field': continue_segmentation_columns})
continue_segmentation_df['Data type'] = 'Continuous'

# Concat both DataFrames
combined_dataframe = pd.concat([segmentation_df, continue_segmentation_df], axis=0).reset_index(drop=True)

reference_dict['s111'] = ('table2',combined_dataframe) 

corpus += f"""\nQuestion: What is the list of segmentation fields?
Answer: The list of segmentation fields is *xgfw|s111|1111.\n"""

corpus += f"""\nQuestion: How is the list of segmentation fields defined?
Answer: The list of segmentation fields is defined as *xgfw|s111|1111.\n"""

corpus += f"""\nQuestion: What is the list of segmentation variables?
Answer: The list of segmentation variables is *xgfw|s111|1111.\n"""

corpus += f"""\nQuestion: How is the list of segmentation variables defined?
Answer: The list of segmentation variables is defined as *xgfw|s111|1111.\n"""

#reference_dict['b111'] = ('list',segmentation_columns) 

#corpus += f"""\nQuestion: What is the list of discrete fields?
#Answer: The list of discrete fields is *xgfw|b111|1111.\n"""

#corpus += f"""\nQuestion: How is the list of discrete fields defined?
#Answer: The list of discrete fields is defined as *xgfw|b111|1111.\n"""

#reference_dict['b112'] = ('list',continue_segmentation_columns) 

#corpus += f"""\nQuestion: What is the list of continuous fields?
#Answer: The list of discrete fields is *xgfw|b112|1111.\n"""

#corpus += f"""\nQuestion: How is the list of continuous fields defined?
#Answer: The list of discrete fields is defined as *xgfw|b112|1111.\n"""


# iterate over segmentation columns

# Create an empty list to store all result DataFrames
all_results = []

for seg_column in segmentation_columns:
    for unique_value in dataset[seg_column].unique():
        subset = dataset[dataset[seg_column] == unique_value]
        result_df = calculate_metrics(subset, kpi_columns, tgcg_column)
        result_df = result_df[result_df['P-value'] <= significance_treshold]
        if not result_df.empty:
            # Add 'Segmentation Column' to result_df
            result_df.insert(0, 'Segmentation Column', seg_column)
            # Add 'value' column to result_df
            result_df.insert(1, 'value', unique_value)
            # Add result_df to all_results list
            all_results.append(result_df)

# Concatenate all the DataFrames in the all_results list
all_results_df = pd.concat(all_results)

# Reset the DataFrame's index to ensure it is unique
all_results_df.reset_index(drop=True, inplace=True)

reference_dict['d111'] = ('table',all_results_df) 

corpus += f"""\nQuestion: What are the discrete fields with significant results?
Answer: The discrete fields with significant results are *xgfw|d111|1111.\n"""

corpus += f"""\nQuestion: Can you provide details about the discrete fields with significant results?
Answer: The details for the discrete fields with significant results are *xgfw|d111|1111.\n"""

corpus += f"""\nQuestion: How are the discrete fields with significant results defined?
Answer: The discrete fields with significant results are defined as *xgfw|d111|1111.\n"""

corpus += f"""\nQuestion: What is the list of discrete variables with significant results?
Answer: The list of discrete variables with significant results is *xgfw|d111|1111.\n"""

corpus += f"""\nQuestion: Can you provide the details of the list of discrete variables with significant results?
Answer: The details for the list of discrete variables with significant results are *xgfw|d111|1111.\n"""

corpus += f"""\nQuestion: How is the list of discrete variables with significant results defined?
Answer: The list of discrete variables with significant results is defined as *xgfw|d111|1111.\n"""



#apply the style and display de df
#st.dataframe(all_results_df.style.apply(highlight_pvalue, axis=1))

##seccion variables segmentacion continuas
#Para calcular los mejores segmentos (intervalos) de variables continuas (sin definir cuantiles ) primero debemos reescalar el control group para tener el mismo número de elementos que target

# Identify the continuous segmentation columns
continuous_segmentation_columns = information_dataset.loc[
    (information_dataset['METATYPE'] == 'SF') & 
    (information_dataset['DATATYPE'].isin(['NUMERIC'])), 'COLUMN'].values

# Identify the minority and majority classes
minority_class = dataset[tgcg_column].value_counts().idxmin()
majority_class = dataset[tgcg_column].value_counts().idxmax()

# Split the dataset into two based on the minority and majority classes
minority_df = dataset[dataset[tgcg_column] == minority_class]
majority_df = dataset[dataset[tgcg_column] == majority_class]

# Oversample the minority class
minority_oversampled = minority_df.sample(len(majority_df), replace=True, random_state=42)

# Combine the oversampled dataframe with the majority class dataframe
oversampled_df = pd.concat([majority_df, minority_oversampled], axis=0)

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=["Segmentation Field", "Lower limit", "Upper limit", "KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"])

# Iterate over the different numerical fields for which we want to calculate the best intervals
for seg_column in continuous_segmentation_columns:
    for kpi in kpi_columns:
        # Create a dataframe with 'target' records only
        target_df = oversampled_df[oversampled_df[tgcg_column] == 'target'][[tgcg_column, seg_column,kpi]]

        #calculate target acceptance (to be used as a penalty in the Kadane's algorithm )
        target_acceptance = target_df[kpi].sum()/len(target_df)

        # Create a dataframe with 'control' records only
        control_df = oversampled_df[oversampled_df[tgcg_column] == 'control'][[tgcg_column, seg_column,kpi]]

        #this penalty is added to avoid segments with a lot of zeroes in the kadane algorithm
        control_df[kpi] = control_df[kpi] +target_acceptance 

        # Sort the dataframes
        target_df.sort_values(by=seg_column, inplace=True)
        control_df.sort_values(by=seg_column, inplace=True)

        # Reset the indices of the dataframes
        target_df.reset_index(drop=True, inplace=True)
        control_df.reset_index(drop=True, inplace=True)

        # Rename the columns in control_df before concatenating
        control_df.columns = [col + "_control" for col in control_df.columns]

        # Concatenate the two dataframes
        concatenated_df = pd.concat([target_df, control_df], axis=1)

        # Calculate the mean of seg_column and seg_column_control, in case some values do not match exactly
        concatenated_df[seg_column] = (concatenated_df[seg_column] + concatenated_df[seg_column + "_control"]) / 2

        # Calculate the difference between kpi and kpi_control
        concatenated_df['granular_uplift'] = concatenated_df[kpi] - concatenated_df[kpi + "_control"]

        # Find the subintervals that maximize the sum of granular_uplift
        granular_uplift_array = concatenated_df['granular_uplift'].values
        max_granular_uplift, start_index, end_index = kadane_algorithm(granular_uplift_array)

        #now we calculate the best intervals for the array in both directions (max and min)
        def process_segmentation(granular_uplift_array, concatenated_df, dataset, seg_column, kpi, tgcg_column, results_df):
            max_granular_uplift, start_index, end_index = kadane_algorithm(granular_uplift_array)
            if start_index > end_index:
                max_granular_uplift, start_index, end_index = kadane_algorithm_mod(granular_uplift_array)

            start_value = concatenated_df.iloc[start_index][seg_column]
            end_value = concatenated_df.iloc[end_index][seg_column]

            filtered_dataset = dataset[(dataset[seg_column] >= start_value) & (dataset[seg_column] <= end_value)]
            
            result_df = calculate_metrics2(filtered_dataset, kpi, tgcg_column)
            
            if not result_df.empty:
                for index, row in result_df.iterrows():
                    new_row = pd.DataFrame({
                        "Segmentation Field": [seg_column],
                        "Lower limit": [start_value],
                        "Upper limit": [end_value],
                        "KPI": [kpi],
                        "TG Acceptors": [row["TG Acceptors"]],
                        "TG Acceptance (%)": [row["TG Acceptance (%)"]],
                        "CG Acceptors": [row["CG Acceptors"]],
                        "CG Acceptance (%)": [row["CG Acceptance (%)"]],
                        "Uplift (%)": [row["Uplift (%)"]],
                        "P-value": [row["P-value"]],
                    })
                    results_df = pd.concat([results_df, new_row], ignore_index=True)
            
            return results_df

        for k in range(2):
            if k == 1:
                granular_uplift_array = get_negative_array(granular_uplift_array)
            
            results_df = process_segmentation(granular_uplift_array, concatenated_df, dataset, seg_column, kpi, tgcg_column, results_df)

results_df["TG Acceptors"] = results_df["TG Acceptors"].astype(float).round(0).astype(int)
results_df["CG Acceptors"] = results_df["CG Acceptors"].astype(float).round(0).astype(int)

#exclude non-significant results
results_df = results_df[results_df['P-value'] <= significance_treshold]

reference_dict['c111'] = ('table',results_df) 

corpus += f"""\nQuestion: What are the continuous fields with significant results?
Answer: The continuous fields with significant results are *xgfw|c111|1111.\n"""

corpus += f"""\nQuestion: Can you provide details about the continuous fields with significant results?
Answer: The details for the continuous fields with significant results are *xgfw|c111|1111.\n"""

corpus += f"""\nQuestion: How are the continuous fields with significant results defined?
Answer: The continuous fields with significant results are defined as *xgfw|c111|1111.\n"""

corpus += f"""\nQuestion: What is the list of continuous variables with significant results?
Answer: The list of continuous variables with significant results is *xgfw|c111|1111.\n"""

corpus += f"""\nQuestion: Can you provide the details of the list of continuous variables with significant results?
Answer: The details for the list of continuous variables with significant results are *xgfw|c111|1111.\n"""

corpus += f"""\nQuestion: How is the list of continuous variables with significant results defined?
Answer: The list of continuous variables with significant results is defined as *xgfw|c111|1111.\n"""



#st.markdown(f"# Best intervals for continuous variables")


#st.dataframe(results_df.style.apply(highlight_pvalue, axis=1))            
#st.markdown(download_csv_link(results_df, f"results_{seg_column}_{unique_value}.csv"), unsafe_allow_html=True)

##fin seccion variables segmentacion continuas

"""

def calculate_relative_uplift(subset, kpi1, kpi2, value1, value2):
    tg_count = len(subset[(subset[tgcg_column] == 'target') & (subset[kpi1] == value1) & (subset[kpi2] == value2)])
    tg_total = len(subset[subset[tgcg_column] == 'target'])

    cg_count = len(subset[(subset[tgcg_column] == 'control') & (subset[kpi1] == value1) & (subset[kpi2] == value2)])
    cg_total = len(subset[subset[tgcg_column] == 'control'])

    tg_acceptance = round((tg_count / tg_total) * 100, 2) if tg_total != 0 else 0
    cg_acceptance = round((cg_count / cg_total) * 100, 2) if cg_total != 0 else 0
    uplift = tg_acceptance - cg_acceptance
    relative_uplift = (uplift / cg_acceptance) * 100 if cg_acceptance != 0 else 0
    return relative_uplift

st.markdown(f"# Cross-KPI results")


# Create an empty DataFrame with KPI labels as indices and columns.
kpi_labels = [f'{kpi} = {val}' for kpi in kpi_columns for val in [0, 1]]
matrix_df = pd.DataFrame(index=kpi_labels, columns=kpi_labels)

# Fill matrix with relative uplifts
for row_label, col_label in itertools.product(kpi_labels, repeat=2):
    kpi1, value1 = row_label.split(' = ')
    kpi2, value2 = col_label.split(' = ')
    matrix_df.loc[row_label, col_label] = calculate_relative_uplift(dataset, kpi1, kpi2, int(value1), int(value2))

# Display the matrix in the user interface
st.write(matrix_df)

# Define colors for the heatmap: green for positive values, red for negative values.
colors = ['red', 'lightgray', 'green']

# First we convert the values to numeric, forcing non-numerics to NaN
values = pd.to_numeric(matrix_df.values.flatten(), errors='coerce').reshape(matrix_df.values.shape)

# Now we replace NaNs with 0
values = np.where(np.isnan(values), 0, values)

# Convert numeric values to strings with two decimal places
text_values = np.round(values, 0).astype(int).astype(str)
# Append '%' to each individual item
text = [f"{val}%" for val in text_values.flatten()]
# Reshape the array
text = np.array(text).reshape(values.shape)

fig = go.Figure(data=go.Heatmap(
    z=values,
    x=matrix_df.columns,
    y=matrix_df.index[::-1],  # Reverse the order of the index for the heatmap
    colorscale=colors,
    zmid=0,
    text=text,
    texttemplate="%{text}",
    textfont={"size": 10},
    hoverongaps = False
))

# Adjust the layout of the figure.
fig.update_layout(
    title='Heatmap of Relative Uplift',
    xaxis_nticks=len(matrix_df.columns),
    yaxis_nticks=len(matrix_df.index)
)

# Display the figure in the Streamlit user interface.
st.plotly_chart(fig)

"""