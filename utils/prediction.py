import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_elements import elements, mui, html
from streamlit_elements import dashboard
from streamlit_elements import nivo
import plotly.graph_objects as go
import numpy as np
import streamlit_toggle as tog
from utils.prediction import *
from utils.radar_utils_2 import *

def normalize_column_names(columns):
    normalized = []
    for col in columns:
        norm_col = col.replace('(ppb)', '').strip()
        normalized.append(norm_col)
    return normalized

def validate_columns(columns, required_columns, min_count):
    normalized_cols = normalize_column_names(columns)
    normalized_req = normalize_column_names(required_columns)
    common_cols = set(normalized_cols).intersection(set(normalized_req))
    return len(common_cols) >= min_count

def rename_columns(df, required_columns):
    col_map = {col.replace('(ppb)', '').strip(): col for col in df.columns}
    normalized_required = normalize_column_names(required_columns)
    new_cols = {col_map[col]: col for col in normalized_required if col in col_map}
    return df.rename(columns=new_cols)

def run():
    TASTES = ['intensity', 'sweetness', 'acidity', 'tannin', 'fizziness']
    MINERALS = ['B', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti',
                'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'As', 'Br', 'Rb', 'Sr',
                'Y', 'Zr', 'Nb', 'Cd', 'Sn', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                'Sm', 'W', 'Tl', 'Pb', 'U']

    CATEGORIES = ['Pays', 'Domaine', 'Cuvée', 'Appellation', "Région viticole", "cepage1", 
                  'certification', 'Type', 'categorie', 'millesime']

    st.set_page_config(
        page_title="M&Wine App",
        page_icon="mandwine_logo.jpeg",
        layout="wide"
    )

    def plot_top_candidates(top_k_groups, target, features, input_vector, normalized_data, MINERALS):
        columns = st.columns(len(top_k_groups))
        for col_index, col in enumerate(columns):
            target_group = top_k_groups[col_index]
            representative_vector = normalized_data[normalized_data[target] == target_group]
            for feature in features:
                representative_vector[feature] = representative_vector[feature].median()
            fig = plot_radar_chart(pd.concat([pd.DataFrame(representative_vector, columns=features), input_vector], axis=0), MINERALS, target_group)
            with col:
                st.plotly_chart(fig, use_container_width=True)

    @st.cache_data
    def load_data(file):
        ext = file.name.split('.')[-1]
        if ext == 'csv':
            return pd.read_csv(file)
        elif ext in ['xls', 'xlsx']:
            return pd.read_excel(file)
        else:
            st.error("Unsupported file format")
            st.stop()

    st.markdown("<h1 style='text-align: center; color: black;'> M & Wine AI Origin</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    with st.expander("See App Description"):
        st.write("")
        st.write("Welcome to **AI-Origin by M & Wine**, the innovative app designed to predict the origins of wine based on mineral concentrations. Developed by M & Wine, our app uses AI to analyze and predict multiple aspects of wine, including category, color, country, region, grape variety, appellation, vintage, and organic status.")
        st.write("By inputting mineral data, users can quickly receive detailed predictions that help in **selecting**, **categorizing**, and **understanding** wines at a deeper level. ")
        st.write("This app is designed to be user-friendly and intuitive, making it easy for wine enthusiasts, sommeliers, et industry professionals to access and use the data they need.")

    st.write("")

    uploaded_file = st.file_uploader("Please upload your input data (CSV or Excel)", type=["csv", "xls", "xlsx"])
    if uploaded_file is None:
        st.warning("Please upload a file")
        st.stop()

    df = load_data(uploaded_file)
    original_db = df.copy()  # Save the original data to use later

    if validate_columns(df.columns, MINERALS, 15):
        df = rename_columns(df, MINERALS)
        st.success("At least 15 necessary minerals are present in the uploaded file")
    else:
        st.error("Please upload a file with at least 15 of the necessary minerals present")
        st.stop()

    # Filter out MINERALS that are not present in the dataframe
    available_minerals = [mineral for mineral in MINERALS if mineral in df.columns]

    st.write("### File Preview:")
    st.dataframe(df.head(10))

    selected_row_index = st.number_input("Select the row number to analyze", min_value=0, max_value=len(df)-1, step=1)
    input_vector = df.iloc[[selected_row_index]].copy()

    with st.expander("What Minerals should be considered in the predictions ? (All by default)"):
        mineral_options = st.multiselect(
            "Customize the minerals to consider in the predictions",
            available_minerals,
            available_minerals)

    available_minerals = mineral_options

    col1, center, col3 = st.columns([1, 3, 1])
    with center:
        fig = plot_radar_chart(input_vector, available_minerals)
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    st.markdown("***")
    st.markdown("<h1 style='text-align: center; color: black; font-size: 30px;'>Characteristics :</h1>", unsafe_allow_html=True)
    st.write("")
    taste_columns = st.columns([1, 1, 1, 1, 1])
    
    df_copy = df.copy()
    df_copy.columns = [col.replace('(ppb)', '').strip() for col in df_copy.columns]  # Normalize the column names

    for column, taste in enumerate(TASTES):
        taste_columns[column].write(f"***{taste}*** : {predict_bottle(df_copy.iloc[[selected_row_index]], taste, available_minerals)}")

    st.write("")
    st.markdown("***")
    st.write("")

    cols1, cols2, cols3, cols4, cols5, cols6, cols7, cols8 = st.columns([2, 1, 1, 1, 1, 1, 1, 1])
    with cols1:
        switch = tog.st_toggle_switch(label="Adaptive Filtering",
                                      key="Key1",
                                      default_value=False,
                                      label_after=False,
                                      inactive_color='#D3D3D3',
                                      active_color="#11567f",
                                      track_color="#29B5E8"
                                      )

    with st.form(key='categories_Selection_form'):
        selections = st.multiselect("Select the categories to predict in the desired order", CATEGORIES)
        st.form_submit_button("Predict")
    if len(selections) == 0:
        st.warning("Please select at least one category")
        st.stop()
    else:
        if switch:
            predictions_dict = {}
        else:
            predictions_dict = None
        for selection in selections:
            st.markdown("<h3 style='text-align: center; color: black;'> Predicting {} </h1>".format(selection), unsafe_allow_html=True)
            top_k_groups, distances = find_top_k_groups(original_db, input_vector, available_minerals, selection, predictions_dict=predictions_dict)
            if top_k_groups is None:
                st.stop()
            plot_top_candidates(top_k_groups, selection, available_minerals, input_vector, original_db, available_minerals)
            model, classification_report, classes = train_classifier(original_db, available_minerals, selection, top_k_groups, predictions_dict)
            if model:
                prediction = model.predict(input_vector[available_minerals].values)
                if switch:
                    predictions_dict[selection] = classes[prediction[0]]
                st.write(f"The predicted {selection} is ***{classes[prediction[0]]}***")
                with st.expander("See more details about model performance"):
                    st.dataframe(
                        pd.DataFrame(
                            classification_report
                        ).transpose()
                        , width=1200)

                st.write("***")
            else:
                st.warning("Not enough Data To Train Model for this case. The most likely group is : " + top_k_groups[0])
                if switch:
                    predictions_dict[selection] = top_k_groups[0]

if __name__ == "__main__":
    run()
