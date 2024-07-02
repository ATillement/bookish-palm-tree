import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import streamlit_toggle as tog
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

from utils.prediction import *
from utils.radar_utils_2 import *
from utils.recommender_utils import *

def run():
    # Load session state
    session_state = st.session_state
    
    TASTES = ['intensity', 'sweetness', 'acidity', 'tannin', 'fizziness']
    MINERALS = ['B', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti',
                'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'As', 'Br', 'Rb', 'Sr',
                'Y', 'Zr', 'Nb', 'Cd', 'Sn', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                'Sm', 'W', 'Tl', 'Pb', 'U']

    CATEGORIES = ['Pays', 'Domaine', 'Cuvée', 'Appellation', "Région viticole", "cepage1", 
                  'certification', 'Type', 'categorie', 'millesime']

    st.set_page_config(
        page_title="Wine Authentication Page",
        page_icon="wine_logo.jpeg",
        layout="wide"
    )

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

    original_db = load_data()

    st.markdown("<h1 style='text-align: center; color: black;'> Wine Authentication Page </h1>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    with st.expander("See App Description"):
        st.write("")
        st.write("Welcome to **Wine Authentication**, the innovative app designed to authenticate wines based on mineral concentrations. Developed by M & Wine, our app uses AI to analyze and predict multiple aspects of wine, including authenticity.")
        st.write("By inputting mineral data, users can quickly receive detailed predictions that help in **selecting**, **categorizing**, and **authenticating** wines at a deeper level.")
        st.write("This app is designed to be user-friendly and intuitive, making it easy for wine enthusiasts, sommeliers, and industry professionals to access and use the data they need.")

    st.write("")

    uploaded_file = st.file_uploader("Please upload your input data (CSV or Excel)", type=["csv", "xls", "xlsx"])
    if 'uploaded_file' in session_state and uploaded_file is None:
        uploaded_file = session_state['uploaded_file']

    if uploaded_file is None:
        st.warning("Please upload a file")
        st.stop()

    df = load_data2(uploaded_file)
    
    if validate_columns(df.columns, MINERALS, 15):
        df = rename_columns(df, MINERALS)
        st.success("At least 15 necessary minerals are present in the uploaded file")
    else:
        st.error("Please upload a file with at least 15 of the necessary minerals present")
        st.stop()

    # Filter out MINERALS that are not present in the dataframe
    available_minerals = [mineral for mineral in MINERALS if mineral in df.columns]
    df = replace_mineral_values(df, available_minerals)
    df = prepare_data_for_app(df, bins_info, minerals=MINERALS)

    st.write("### File Preview:")
    st.dataframe(df, height=200)  # Set the height to enable scrolling
    input_vector = 0
    with st.form(key='row_selection_form'):
        selected_row_index = st.number_input("Select the row number to analyze", min_value=0, max_value=len(df)-1, step=1)
        submit_button = st.form_submit_button(label='Select')
        input_vector = df.iloc[[selected_row_index]].copy()
        session_state['input_vector'] = input_vector
        session_state['selected_row_index'] = selected_row_index
        session_state['uploaded_file'] = uploaded_file

    def update_options_region():
        st.session_state.options_region = original_db['Région viticole'].dropna().unique().tolist()

    def update_options_grape():
        st.session_state.options_grape = original_db['cepage1'].dropna().unique().tolist()

    if 'options_region' not in st.session_state:
        update_options_region()

    if 'options_grape' not in st.session_state:
        update_options_grape()

    with st.expander('Database Selection'):
        country_filter = st.selectbox("Choose the country filter", ["All Countries", "French Only", "Exclude France"])
        wine_color = st.selectbox("Choose the wine color", ["All", "Rouge", "Blanc", "Rosé"])
        wine_type = st.selectbox("Choose the wine type", ["All", "Tranquille", "Effervescent"])

        wine_region_choice = st.selectbox(
            "Choose the wine region",
            ["All", "Select from list"] + st.session_state.options_region,
            on_change=update_options_region
        )
        if wine_region_choice == "Select from list":
            wine_region = st.selectbox("Select the region", st.session_state.options_region)
        else:
            wine_region = wine_region_choice

        wine_grapes_choice = st.selectbox(
            "Choose the wine grapes",
            ["All", "Select from list"] + st.session_state.options_grape,
            on_change=update_options_grape
        )
        if wine_grapes_choice == "Select from list":
            wine_grapes = st.selectbox("Select the grape variety", st.session_state.options_grape)
        else:
            wine_grapes = wine_grapes_choice

        # Apply filters to original_db
        filtered_db = original_db.copy()
        if country_filter != "All Countries":
            if country_filter == "French Only":
                filtered_db = filtered_db[filtered_db['Pays'] == 'France']
            elif country_filter == "Exclude France":
                filtered_db = filtered_db[filtered_db['Pays'] != 'France']
        if wine_color != "All":
            filtered_db = filtered_db[filtered_db['Type'] == wine_color]
        if wine_type != "All":
            filtered_db = filtered_db[filtered_db['categorie'] == wine_type]
        if wine_region != "All":
            filtered_db = filtered_db[filtered_db['Région viticole'] == wine_region]
        if wine_grapes != "All":
            filtered_db = filtered_db[filtered_db['cepage1'] == wine_grapes]

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

        if use_top_features == "Use top features for best results":
            feature_importance_df = calculate_feature_importance(filtered_db, MINERALS, 'Pays')
            available_minerals = get_top_features(feature_importance_df)

    if 'category_authentication' not in st.session_state:
        st.session_state.category_authentication = CATEGORIES[0]

    def update_options():
        st.session_state.options = filtered_db[st.session_state.category_authentication].unique()

    # Set the default options
    if 'options' not in st.session_state:
        update_options()

    with st.expander("What Minerals should be considered in the predictions ? (All by default)"):
        category_authentication = st.selectbox(
            'Authentication Aspect',
            CATEGORIES,
            key='category_authentication',
            on_change=update_options
        )

        if st.session_state.category_authentication:
            option_selected = st.selectbox(
                'Specific Aspect to Authenticate',
                st.session_state.options
            )

        submit_button = st.button('Authenticate')

    if submit_button:
        # Ensure available_minerals only includes columns present in both original_db and input_vector
        available_minerals = [mineral for mineral in MINERALS if mineral in filtered_db.columns and mineral in input_vector.columns]

        column = st.session_state.category_authentication
        testing_value = option_selected

        data_positive = filtered_db[filtered_db[column] == testing_value]
        data_negative = filtered_db[filtered_db[column] != testing_value]
        if len(data_negative) < 50 or len(data_positive) < 50:
            st.warning("Not enough samples against the category to test to be trained on. \n Please select another category or option.\n \
                        The minimum number of samples should be 50 for each category. \n \
                        The current number of samples for the selected category is {}.".format(len(data_positive)))
            st.stop()
        data_negative[column] = data_negative[column].apply(lambda x: 'not ' + testing_value)

        min_len = min(len(data_positive), len(data_negative))

        data_positive_sliced = data_positive.sample(n=min_len, random_state=42)
        data_negative_sliced = data_negative.sample(n=min_len, random_state=42)

        new_data = pd.concat([data_positive_sliced, data_negative_sliced], ignore_index=True)

        # Perform binary classification
        features = available_minerals
        target = column

        X = new_data[features]
        y = new_data[target]
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        
        # Prediction on the input vector
        prediction_prob = model.predict_proba(input_vector[features].values)[0]
        prediction = model.predict(input_vector[features].values)
        
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        predicted_prob = prediction_prob[prediction.argmax()]

        if predicted_label == testing_value:
            st.success(f"Our Authentication Model Predicts ***{predicted_label}*** as a {column} with a probability of {predicted_prob:.2f}")
        else:
            st.error(f"Our Authentication Model Predicts ***{predicted_label}*** as a {column} with a probability of {predicted_prob:.2f}")

        with st.expander("See more details about authentication model performance"):
            st.dataframe(
                pd.DataFrame(report).transpose(),
                width=1000
            )

        # Radar Plot Visualization
        st.write("")
        st.markdown("***")
        st.markdown("<h1 style='text-align: center; color: black;'>Radar Chart of the Input Sample</h1>", unsafe_allow_html=True)
        fig = plot_radar_chart(input_vector, available_minerals)
        st.plotly_chart(fig, use_container_width=True)

        # Display Confusion Matrix
        st.write("")
        st.markdown("***")
        st.markdown("<h1 style='text-align: center; color: black;'>Confusion Matrix</h1>", unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred, labels=label_encoder.transform(label_encoder.classes_))
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        cm_display.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
        st.pyplot(plt.gcf())


        # ROC Curve Visualization
        st.write("")
        st.markdown("***")
        st.markdown("<h1 style='text-align: center; color: black;'>ROC Curve</h1>", unsafe_allow_html=True)
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        fig_roc = plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(fig_roc)

if __name__ == "__main__":
    run()
