import streamlit as st
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import plot_tree
from app_functions import *

# dataset and information_dataset
dataset = st.session_state.uploaded_data
information_dataset = st.session_state.user_defined_info_dataset

corpus = "" #corpus doc for the chatbot
reference_dict = {} #reference dict to extract python objects

# Extract the TGCG column from the dataset
tgcg_column = information_dataset.loc[information_dataset['METATYPE'] == 'TGCG', 'COLUMN'].values[0]

# Set to lower TGCG column: TARGET-> target, Control -> control
dataset[tgcg_column] = dataset[tgcg_column].str.lower()

# Create new column with tgcg as flags
dataset['tgcg_fl'] = np.where(dataset[tgcg_column] == 'target', 1, 0)

#we need to iterate over each kpi

# Get all KPI columns
kpi_columns = information_dataset.loc[information_dataset['METATYPE'] == 'KPI', 'COLUMN'].values
kpi_columns = kpi_columns.tolist()
# 1. Identify the segmentation columns

segmentation_columns = information_dataset.loc[(information_dataset['METATYPE'] == 'SF') & 
                                              (information_dataset['DATATYPE'].isin(['NUM_ST', 'BOOL', 'STRING'])), 
                                              'COLUMN'].values
continue_segmentation_columns = information_dataset.loc[(information_dataset['METATYPE'] == 'SF') & 
                                              (information_dataset['DATATYPE'].isin(['NUMERIC'])), 
                                              'COLUMN'].values
#convert category columns to category type in pandas
for col in segmentation_columns:
    dataset[col] = dataset[col].astype('category')

dataset_fix = dataset.copy() #store a copy of the dataset

# Conditions before processing
num_records = dataset.shape[0]
num_control_records = dataset[dataset['tgcg_fl'] == 0].shape[0]
num_target_records = dataset[dataset['tgcg_fl'] == 1].shape[0]

if num_records > 2000000:
    st.write("The number of total records is greater than 2,000,000. The model cannot be processed.")
elif num_records < 200:
    st.write("The number of total records is less than 200. The model cannot be processed.")
elif num_control_records < 100:
    st.write("There are fewer than 100 control records. The model cannot be processed.")
elif num_target_records < 100:
    st.write("There are fewer than 100 target records. The model cannot be processed.")
else:
    for kpi in kpi_columns:
        #st.write(kpi_columns)
        #st.header(f"KPI: {kpi}")
        dataset = dataset_fix.copy() #after the first iteration we need to use a clean version of the dataset
        dataset_copy = dataset.copy()
        kpi_original = kpi
        #create model_target column ()
        dataset_copy[kpi + '_model_target'] = dataset_copy.apply(lambda row: 1 if row['tgcg_fl'] == row[kpi] else 0, axis=1)
        kpi = kpi + '_model_target'
        # Create a list with all the columns of interest
        all_columns = list(segmentation_columns) + list(continue_segmentation_columns) + ['tgcg_fl'] + [kpi]

        # Create a new DataFrame that only contains the columns of interest
        df_subset = dataset_copy[all_columns]

        # Apply the oversample function to the new DataFrame
        df_oversampled = oversample(df_subset, [kpi, 'tgcg_fl'])

        #train model
        # Definir las características (X) y la variable objetivo (y)
        X = df_oversampled.drop(columns=['tgcg_fl', kpi])
        y = df_oversampled[kpi]

        # Dividir los datos en conjuntos de entrenamiento y prueba
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # train model
        model = LGBMRegressor(max_depth = 5, n_estimators = 50,  min_child_samples =50) #ponemos un número relativamente alto de min_child_samples porque al sobremuestrear nuestro dataset crece mucho
        model.fit(X, y)

        # Hacer predicciones en el conjunto de prueba
        #y_pred = model.predict(X_test)

        # Create a new list of columns that only includes columns present in 'dataset'
        dataset_columns = list(segmentation_columns) + list(continue_segmentation_columns)

        # Define the features (X) for the original dataset
        X_dataset = dataset[dataset_columns]

        # Make predictions on the original dataset
        dataset_predictions = model.predict(X_dataset)

        # predictions added as a new column in your original DataFrame
        dataset['predictions'] = dataset_predictions

        # Remove "_model_target" from the kpi variable and set the title of the plot
        kpi_name = kpi.replace("_model_target", "")
        #st.header(f"Model scores of the kpi: {kpi_name}")
        #st.write(dataset.drop(columns = ['tgcg_fl']))


        # After creating the predictions column in the original dataset
        # Define the 25th and 75th percentiles
        upper_quartile = np.percentile(dataset['predictions'], 75)
        lower_quartile = np.percentile(dataset['predictions'], 25)

        # Create a new binary column where 'Top25%' indicates the prediction is above the upper quartile
        # and 'Bottom25%' indicates the prediction is below the lower quartile
        dataset['top_bottom'] = np.where(dataset['predictions'] >= upper_quartile, 'Top25%', np.where(dataset['predictions'] <= lower_quartile, 'Bottom25%', np.nan))

        # Create an empty DataFrame to store the results
        results_df = pd.DataFrame(columns=["KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"])

        ### *** Explanining the lgbm using a simple decision tree *** ###

        # Get the feature importance as an array
        importance = model.feature_importances_

        # Create a DataFrame to represent feature importance
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,  # Feature names
            'Importance': importance  # Feature importance
        })

        # we can sort the DataFrame by feature importance
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Identify the top 7 most important features
        top7_features = feature_importance_df['Feature'].head(7)

        # Remove rows where top_bottom is NaN (i.e., predictions not in the top or bottom quartile)
        dataset_binary = dataset.dropna(subset=['top_bottom'])


        # Define the features (X) and the target variable (y)
        X_binary = dataset_binary[top7_features]
        y_binary = dataset_binary['top_bottom']

        # Perform one-hot encoding on the categorical features
        X_binary_encoded = pd.get_dummies(X_binary, prefix_sep='==')

        # Train the decision tree with only the top 7 most important features
        dt_model = DecisionTreeClassifier(max_depth=4)  # fix max depth
        dt_model.fit(X_binary_encoded, y_binary)


    ###################################
        #explain tree rules
        # Use the function get_rules to extract dt rules

        rules_top25 = get_rules(dt_model, X_binary_encoded.columns, dt_model.classes_, 'Top25%')
        rules_Bottom25 = get_rules(dt_model, X_binary_encoded.columns, dt_model.classes_, 'Bottom25%')

        # Display the rules in Streamlit with modified background color
        #st.markdown("\n\n**Best subgroups identified**")

        #now we want to measure the uplift of the subset defined by the dt (for both bottom and top)
        # Select the same features as used for the model and apply the same transformations
        X_original = dataset[top7_features]
        X_original_encoded = pd.get_dummies(X_original, prefix_sep='==')

        # Use the trained model to predict the classes
        dataset['dt_classification'] = dt_model.predict(X_original_encoded)


        # Create an empty DataFrame to store the results
        results_df = pd.DataFrame(columns=[ "KPI", "TG Acceptors","TG Acceptance (%)", "CG Acceptors","CG Acceptance (%)", "Uplift (%)", "P-value"])

        top25_dataset = dataset.loc[dataset['dt_classification'] == 'Top25%']
        top25_result_df = calculate_metrics2(top25_dataset, kpi_original, tgcg_column)
        if not top25_result_df.empty:
            for index, row in top25_result_df.iterrows():
                new_row = pd.DataFrame({
                    "KPI": [kpi_original],
                    "TG Acceptors": [row["TG Acceptors"]],
                    "TG Acceptance (%)": [row["TG Acceptance (%)"]],
                    "CG Acceptors": [row["CG Acceptors"]],
                    "CG Acceptance (%)": [row["CG Acceptance (%)"]],
                    "Uplift (%)": [row["Uplift (%)"]],
                    "P-value": [row["P-value"]],
                })
                top25_results_df = pd.concat([results_df, new_row], ignore_index=True)

        top25_results_df["TG Acceptors"] = top25_results_df["TG Acceptors"].astype(float).round(0).astype(int)
        top25_results_df["CG Acceptors"] = top25_results_df["CG Acceptors"].astype(float).round(0).astype(int)        
        
        bottom25_dataset = dataset.loc[dataset['dt_classification'] == 'Bottom25%']
        bottom25_result_df = calculate_metrics2(bottom25_dataset, kpi_original, tgcg_column)
        if not bottom25_result_df.empty:
            for index, row in bottom25_result_df.iterrows():
                new_row = pd.DataFrame({
                    "KPI": [kpi_original],
                    "TG Acceptors": [row["TG Acceptors"]],
                    "TG Acceptance (%)": [row["TG Acceptance (%)"]],
                    "CG Acceptors": [row["CG Acceptors"]],
                    "CG Acceptance (%)": [row["CG Acceptance (%)"]],
                    "Uplift (%)": [row["Uplift (%)"]],
                    "P-value": [row["P-value"]],
                })
                bottom25_results_df = pd.concat([results_df, new_row], ignore_index=True)

        bottom25_results_df["TG Acceptors"] = bottom25_results_df["TG Acceptors"].astype(float).round(0).astype(int)
        bottom25_results_df["CG Acceptors"] = bottom25_results_df["CG Acceptors"].astype(float).round(0).astype(int)

        #st.markdown(f"**Results on the best and worst subgroups**")
        #st.dataframe(top25_results_df.style.apply(highlight_pvalue, axis=1))  
        #st.dataframe(bottom25_results_df.style.apply(highlight_pvalue, axis=1))  

        key_b = '*xgfw|m' + str(111 + kpi_columns.index(kpi_original)) + '|' + str(1111 + kpi_columns.index(kpi_original))
        key_b2 = 'm' + str(111 + kpi_columns.index(kpi_original))
        reference_dict[key_b2] = ('box_top',(rules_top25,top25_results_df))

        corpus += f"""\nQuestion: What are the best segments for the kpi {kpi_original}?
        Answer: The best segments for kpi {kpi_original} are {key_b}.\n"""

        key_w = '*xgfw|n' + str(111 + kpi_columns.index(kpi_original)) + '|' + str(1111 + kpi_columns.index(kpi_original))
        key_w2 = 'n' + str(111 + kpi_columns.index(kpi_original))
        reference_dict[key_w2] = ('box_bottom',(rules_Bottom25, bottom25_results_df))

        corpus += f"""\nQuestion: What are the worst segments for the kpi {kpi_original}?
        Answer: The worst segments for kpi {kpi_original} are {key_w}.\n"""
