import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from utils.radar_utils_2 import *
from utils.recommender_utils import *

MINERALS = ['B', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
            'Co', 'Ni', 'Cu', 'Zn', 'As', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Cd', 'Sn', 'I', 'Cs', 'Ba', 'La',
            'Ce', 'Pr', 'Nd', 'Sm', 'W', 'Tl', 'Pb', 'U']

CATEGORIES = ['Pays', 'Domaine', 'Cuvée', 'Appellation', "Région viticole", "cepage1", 'certification', 'Type', 'millesime']

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
        pca = joblib.load('./models/pca_model.pkl')
        data = pd.read_csv('./data/data_nor.csv')
        return pca, data

def is_valid_coordinate(value):
    try:
        lat, lon = map(float, value.split(','))
        return True
    except:
        return False

def extract_coordinates(df, geo_column='Geolocalisation'):
    # Filter valid coordinates
    valid_geo = df[geo_column].apply(is_valid_coordinate)
    df = df[valid_geo]

    # Split the coordinates into two columns
    coordinates = df[geo_column].str.split(',', expand=True).astype(float)
    coordinates.columns = ['latitude', 'longitude']
    return coordinates

def main():
    session_state = st.session_state
    pca, original_db = load_data()
    k = 1000

    st.markdown("<h1 style='text-align: center; color: black;'>Wine Recommender System</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("")

    uploaded_file = st.file_uploader("Please upload your input data (minerals vector)", type=["csv", "xls", "xlsx"])
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

    # Filter out MINERALS that are not present in the dataframe
    available_minerals = [mineral for mineral in MINERALS if mineral in df.columns]
    df = replace_mineral_values(df, available_minerals)
    df = prepare_data_for_app(df, bins_info, minerals=available_minerals)  # Notez que nous passons maintenant available_minerals

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
   
    unique_grapes = original_db['cepage1'].dropna().astype(str).unique()
    unique_region = original_db['Région viticole'].dropna().astype(str).unique()

    with st.expander('Database Selection'):
        country_filter = st.selectbox("Choose the country filter", ["All Countries", "French Only", "Exclude France"])
        wine_color = st.selectbox("Choose the wine color", ["All", "Rouge", "Blanc", "Rosé"])
        wine_type = st.selectbox("Choose the wine type", ["All", "Tranquille", "Effervescent"])
    
        wine_region = st.multiselect("Choose specific wine regions", unique_region)
        wine_grapes = st.multiselect("Choose specific wine grapes", unique_grapes)

        if not wine_region:
            wine_region = unique_region.tolist()

        if not wine_grapes:
            wine_grapes = unique_grapes.tolist()
        

    with st.form(key='parameters_selection_form'):
        k_text_input = st.text_input("Enter the number of neighbors to be studied:", k)
        if k_text_input:
            k = int(k_text_input)

        columns = st.multiselect("Select the categories to be studied within the neighbors", CATEGORIES)
        inspect_button = st.form_submit_button("Inspect")
        mean_location_button = st.form_submit_button("Mean localisation")

    if inspect_button:
        if len(columns) == 0:
            columns = ["Région viticole"]
        filter_db = filter_data(original_db, country_filter, wine_color, wine_type, wine_region, wine_grapes)
        df_combined = perform_pca(filter_db, pca, columns)
        if input_vector is not None:
            input_vector_pca = perform_pca(input_vector, pca, ['Nom échantillon'], unknown=True)
        else:
            st.warning("Please select a row to analyze.")
        df_combined = pd.concat([df_combined, input_vector_pca], ignore_index=True)
        center_x, center_y, center_z = compute_center_coordinates(input_vector_pca)
        distances_squared = compute_distances(df_combined, center_x, center_y, center_z)
        (x_sphere, y_sphere, z_sphere), radius, triangles = compute_sphere_coordinates(distances_squared, center_x, center_y, center_z, k=k)
        inside_sphere = distances_squared <= radius**2
        colors = np.where(inside_sphere, 'green', 'rgb(153, 204, 255)')
        fig = plot_figure(triangles, (x_sphere, y_sphere, z_sphere), (center_x, center_y, center_z), colors, df_combined, columns)
        st.plotly_chart(fig, use_container_width=True)
        st.write("")
        st.write("")
        st.markdown("***")
        st.markdown("<h2 style='text-align: center; color: black;'>Statistics of the Minerally Closest Wines</h2>", unsafe_allow_html=True)
        neighbors = df_combined[inside_sphere].copy()
    
        for category in columns:
            _, center, _ = st.columns([1, 3, 1])
            st.markdown("<h3 style='text-align: center; color: black;'>{} Analysis</h3>".format(category), unsafe_allow_html=True)
            plot_statistics(neighbors, category, k, filter_db)
            st.markdown("***")
            st.write("")
            
    if mean_location_button:
            if len(columns) == 0:
                columns = ["Région viticole"]
            filter_db = filter_data(original_db, country_filter, wine_color, wine_type, wine_region, wine_grapes)
            df_combined = perform_pca(filter_db, pca, columns)
            if input_vector is not None:
                input_vector_pca = perform_pca(input_vector, pca, ['Nom échantillon'], unknown=True)
            else:
                st.warning("Please select a row to analyze.")
            df_combined = pd.concat([df_combined, input_vector_pca], ignore_index=True)
            center_x, center_y, center_z = compute_center_coordinates(input_vector_pca)
            distances_squared = compute_distances(df_combined, center_x, center_y, center_z)
            (x_sphere, y_sphere, z_sphere), radius, triangles = compute_sphere_coordinates(distances_squared, center_x, center_y, center_z, k=k)
            inside_sphere = distances_squared <= radius**2
            neighbors = df_combined[inside_sphere].copy()
            if 'Geolocalisation' in filter_db.columns:
                neighbors['Geolocalisation'] = filter_db['Geolocalisation']
                geo_coordinates = extract_coordinates(neighbors)
                mean_latitude = geo_coordinates['latitude'].mean()
                mean_longitude = geo_coordinates['longitude'].mean()

                geo_coordinates['count'] = geo_coordinates.groupby(['latitude', 'longitude']).transform('size')
                geo_coordinates = geo_coordinates.drop_duplicates(subset=['latitude', 'longitude'])

                st.markdown("<h2 style='text-align: center; color: black;'>Average Location of the Closest Wines</h2>", unsafe_allow_html=True)
                hover_data = ['latitude', 'longitude'] + columns
                for col in columns:
                    neighbors[col] = original_db[col]
                    geo_coordinates[col] = neighbors[col]

                st.markdown("<h2 style='text-align: center; color: black;'>Average Location of the Closest Wines</h2>", unsafe_allow_html=True)
                fig = px.scatter_geo(geo_coordinates, lat='latitude', lon='longitude', size='count', scope="world",
                                    title="Average Location of Closest Wines", projection="natural earth", size_max=20,
                                    hover_data=hover_data)
                fig.add_scattergeo(lat=[mean_latitude], lon=[mean_longitude], 
                                marker=dict(size=10, color='red', symbol='star'), 
                                name="Mean Location")
                st.plotly_chart(fig, use_container_width=True)
        

if __name__ == "__main__":
    main()
