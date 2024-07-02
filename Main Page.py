import streamlit as st
import pandas as pd
from streamlit_elements import elements, mui, html, dashboard, nivo
import plotly.graph_objects as go
import numpy as np
import streamlit_toggle as tog
import pickle
import plotly.colors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from utils.prediction import *
from utils.radar_utils_2 import *
from utils.recommender_utils import *

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
    session_state = st.session_state

    @st.cache_data
    def load_data2(file):
        ext = file.name.split('.')[-1]
        if ext == 'csv':
            return pd.read_csv(file)
        elif ext in ['xls', 'xlsx']:
            return pd.read_excel(file)
        else:
            st.error("Unsupported file format")
            st.stop() 
    
    def load_data():
        with st.spinner("Loading data..."):
            return pd.read_csv('./data/data_nor.csv')

    def calculate_feature_importance(data, features, target):
        X = data[features]
        y = data[target]

        X = X.fillna(0)
        y = y.fillna("Unknown")
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
    
        importances = clf.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        return feature_importance_df.sort_values(by='Importance', ascending=False)
    
    def get_top_features(feature_importance_df, threshold=0.67):
        sorted_features = feature_importance_df.sort_values(by='Importance', ascending=False)
        return sorted_features.head(int(len(sorted_features) * threshold))['Feature'].tolist()

    original_db = load_data()    

    st.markdown("<h1 style='text-align: center; color: black;'> M & Wine AI Origin</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    with st.expander("See App Description"):
        st.write("")
        st.write("Welcome to **AI-Origin by M & Wine**, the innovative app designed to predict the origins of wine based on mineral concentrations. Developed by M & Wine, our app uses AI to analyze and predict multiple aspects of wine, including category, color, country, region, grape variety, appellation, vintage, and organic status.")
        st.write("By inputting mineral data, users can quickly receive detailed predictions that help in **selecting**, **categorizing**, and **understanding** wines at a deeper level. ")
        st.write("This app is designed to be user-friendly and intuitive, making it easy for wine enthusiasts, sommeliers, and industry professionals to access and use the data they need.")

    st.write("")

    uploaded_file = st.file_uploader("Please upload your input data (CSV or Excel)", type=["csv", "xls", "xlsx"])
    if 'uploaded_file' in session_state and uploaded_file is None:
        uploaded_file = session_state['uploaded_file']

    if uploaded_file is None:
        st.warning("Please upload a file")
        st.stop()

    df = load_data2(uploaded_file)
    if 'input_vector' in session_state:
        input_vector = session_state['input_vector']
    else:
        input_vector = None

    if validate_columns(df.columns, MINERALS, 15):
        df = rename_columns(df, MINERALS)
        st.success("At least 15 necessary minerals are present in the uploaded file")
    else:
        st.error("Please upload a file with at least 15 of the necessary minerals present")
        st.stop()

    available_minerals = [mineral for mineral in MINERALS if mineral in df.columns]
    df = replace_mineral_values(df, available_minerals)
    df = prepare_data_for_app(df, bins_info, minerals=MINERALS)

    st.write("### File Preview:")
    st.dataframe(df, height=200)
    
    with st.form(key='row_selection_form'):
        selected_row_index = st.number_input("Select the row number to analyze", min_value=0, max_value=len(df)-1, step=1)
        submit_button = st.form_submit_button(label='Select')
        input_vector = df.iloc[[selected_row_index]].copy()
        session_state['input_vector'] = input_vector
        session_state['selected_row_index'] = selected_row_index
        session_state['uploaded_file'] = uploaded_file

    unique_grapes = original_db['cepage1'].dropna().astype(str).unique()
    unique_region = original_db['Région viticole'].dropna().astype(str).unique()

    with st.expander('Database Selection'):
        country_filter = st.selectbox("Choose the country filter", ["All Countries", "French Only", "Exclude France"])
        wine_color = st.selectbox("Choose the wine color", ["All", "Rouge", "Blanc", "Rosé"])
        wine_type = st.selectbox("Choose the wine type", ["All", "Tranquille", "Effervescent"])
    
        wine_region = st.multiselect("Choose specific wine regions", unique_region)
        wine_grapes = st.multiselect("Choose specific wine grapes", unique_grapes)
    
    with st.expander('Parameters Selection'):
        mineral_options = st.multiselect(
            "Customize the minerals to consider in the predictions",
            available_minerals,
            available_minerals
        )

        available_minerals = mineral_options

        use_top_features = st.radio(
            "Choose feature selection method",
            ("Use all features", "Use top features for best results")
        )
        selections =[]
        if use_top_features == "Use top features for best results":
            if len(selections) == 0:
                feature_importance_df = calculate_feature_importance(original_db, MINERALS, 'Pays')
                available_minerals = get_top_features(feature_importance_df)
            elif len(selections) >= 1:
                feature_importance_df = calculate_feature_importance(original_db, MINERALS, selections[0])
                available_minerals = get_top_features(feature_importance_df)

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
    df_copy.columns = [col.replace('(ppb)', '').strip() for col in df_copy.columns]

    st.write("")
    st.markdown("***")
    st.write("")

    cols1, cols2, cols3, cols4, cols5, cols6, cols7, cols8 = st.columns([2, 1, 1, 1, 1, 1, 1, 1])
    with cols1:
        switch = tog.st_toggle_switch(label="Adaptive Filtering", key="Key1", default_value=False, label_after=False,
                                      inactive_color='#D3D3D3', active_color="#11567f", track_color="#29B5E8")

    original_db = filter_data(original_db, country_filter, wine_color, wine_type, wine_region, wine_grapes)

    with st.form(key='categories_Selection_form'):
        selections = st.multiselect("Select the categories to predict in the desired order", CATEGORIES)
        st.form_submit_button("Predict")

    results_summary = []  # To collect results for the summary

    if len(selections) == 0:
        st.warning("Please select at least one category")
        st.stop()
    else:
        if switch:
            predictions_dict = {}
        else:
            predictions_dict = None

        for selection in selections:
            st.markdown("<h3 style='text-align: center; color: black;'> Predicting {} </h3>".format(selection), unsafe_allow_html=True)
            top_k_groups, distances = find_top_k_groups(original_db, input_vector, available_minerals, selection, predictions_dict=predictions_dict)
            
            if top_k_groups is None or len(top_k_groups) == 0:
                st.warning(f"No valid groups found for {selection}. Please try with different parameters or input data.")
                continue

            if selection in ['Appellation', 'Domaine']:
                plot_top_candidates(top_k_groups, distances, selection, available_minerals, input_vector, original_db, available_minerals)
                
                closest_group = distances.sort_values(by='Average Distance').iloc[0][selection]
                st.markdown(f"### The predicted group for this sample is **{closest_group}**")
                results_summary.append(f"Predicted {selection}: **{closest_group}**")
            else:
                plot_top_candidates(top_k_groups, distances, selection, available_minerals, input_vector, original_db, available_minerals)
                
                model, classification_report, classes = train_classifier(original_db, available_minerals, selection, top_k_groups, distances, predictions_dict)
                if model:
                    predicted_class, probability, reliability = display_prediction_results(model, input_vector, available_minerals, classes, original_db)
                    if switch:
                        predictions_dict[selection] = predicted_class
                    results_summary.append(f"Predicted {selection}: **{predicted_class}** with reliability: **{reliability}**")
                    with st.expander("See more details about model performance"):
                        st.dataframe(pd.DataFrame(classification_report).transpose(), width=1200)
                    st.write("***")
                else:
                    if len(top_k_groups) == 1:
                        st.warning("The predicted group for this sample is : " + top_k_groups[0])
                        results_summary.append(f"Predicted {selection}: **{top_k_groups[0]}**")
                    else:
                        st.warning("Not enough Data To Train Model for this case. The most likely group is : " + top_k_groups[0])
                        results_summary.append(f"Most likely {selection}: **{top_k_groups[0]}**")
                    if switch:
                        predictions_dict[selection] = top_k_groups[0]

        st.write("### Prediction Summary:")
        for result in results_summary:
            st.markdown(result)

if __name__ == "__main__":
    run()
