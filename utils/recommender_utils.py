import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
import joblib
import os
import streamlit as st

MINERALS = ['B', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
            'Co', 'Ni', 'Cu', 'Zn', 'As', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Cd', 'Sn', 'I', 'Cs', 'Ba', 'La',
            'Ce', 'Pr', 'Nd', 'Sm', 'W', 'Tl', 'Pb', 'U']

def perform_pca(db, pca, columns, unknown=False):
    if db is None:
        raise ValueError("The input dataframe (db) is None.")
    
    if not isinstance(db, pd.DataFrame):
        raise TypeError("The input (db) should be a pandas DataFrame.")
    
    required_minerals = pca.feature_names_in_

    # Add missing columns with default values (0)
    for mineral in required_minerals:
        if mineral not in db.columns:
            db[mineral] = 0

    if unknown:
        new_vector_transformed = pca.transform(db[required_minerals])
        new_vector_df = pd.DataFrame(new_vector_transformed, columns=['PC1', 'PC2', 'PC3'])
        new_vector_df['Nom échantillon'] = "Unknown"
        return new_vector_df

    df_minerals = db[required_minerals]
    components = pca.transform(df_minerals)
    df_pca = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3'])
    df_combined = pd.concat([db[columns], df_pca], axis=1)
    return df_combined
def compute_center_coordinates(inspection_vector):
    center_x = inspection_vector['PC1'][0]
    center_y = inspection_vector['PC2'][0]
    center_z = inspection_vector['PC3'][0]
    return center_x, center_y, center_z

def compute_distances(df, center_x, center_y, center_z):
    distances_squared = (df['PC1'] - center_x)**2 + (df['PC2'] - center_y)**2 + (df['PC3'] - center_z)**2
    return distances_squared

def plot_figure(triangles, spheres_coordinates, center_coordinates, colors, df_combined, columns):
    center_x, center_y, center_z = center_coordinates
    x_sphere, y_sphere, z_sphere = spheres_coordinates
    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        i=[t[0] for t in triangles],
        j=[t[1] for t in triangles],
        k=[t[2] for t in triangles],
        opacity=0.5,
        color='lightpink',
        name='Highlight Sphere',
        text=df_combined[columns],
        hovertemplate='%{text}',
    ))

    fig.add_trace(go.Scatter3d(
        x=df_combined['PC1'],
        y=df_combined['PC2'],
        z=df_combined['PC3'],
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            opacity=0.8,
            line=dict(
                color='DarkSlateGrey',
                width=0.5
            )
        ),
        text=df_combined[columns],
        hovertemplate='%{text}',
        name='M&Wine DB'
    ))
    fig.add_trace(go.Scatter3d(
        x=[center_x],
        y=[center_y],
        z=[center_z],
        mode='markers',
        marker=dict(
            size=10,
            color='blue'
        ),
        text='New Sample',
        name='New Sample'
    ))

    fig.update_layout(
        title={
            'text': '3D Neighbors Visualization of the input sample',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
        ),
        width=1300,
        height=800
    )
    return fig

def compute_sphere_coordinates(distances_squared, center_x, center_y, center_z, k=1000):
    sorted_distances = np.sort(distances_squared)
    radius = np.sqrt(sorted_distances[k-1])

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    triangles = []

    for i in range(len(u) - 1):
        for j in range(len(v) - 1):
            triangles.append([i * len(v) + j, i * len(v) + (j + 1), (i + 1) * len(v) + j])
            triangles.append([i * len(v) + (j + 1), (i + 1) * len(v) + (j + 1), (i + 1) * len(v) + j])

    x_sphere = center_x + radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = center_y + radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = center_z + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    x_sphere = x_sphere.flatten()
    y_sphere = y_sphere.flatten()
    z_sphere = z_sphere.flatten()
    return (x_sphere, y_sphere, z_sphere), radius, triangles

def filter_data(df, country_filter, wine_color, wine_type,wine_region,wine_grapes):
    if country_filter == "French Only":
        df = df[df['Pays'] == 'France']
    elif country_filter == "Exclude France":
        df = df[df['Pays'] != 'France']

    if wine_color != "All":
        df = df[df['couleur'] == wine_color]

    if wine_type != "All":
        df = df[df['categorie'] == wine_type]
    
    if wine_region:
        df = df[df['Région viticole'].isin(wine_region)]
    
    if wine_grapes:
        df = df[df['cepage1'].isin(wine_grapes)]

    return df



def plot_statistics(neighbors, category, k, filtered_db):
    col1, col2 = st.columns([0.6, 0.4])
    value_counts = neighbors[category].value_counts().reset_index()
    value_counts.columns = [category, 'count']
    
    total_counts = filtered_db[category].value_counts().reset_index()
    total_counts.columns = [category, 'total_count']

    # Merge with total counts to get proportions
    merged_counts = pd.merge(value_counts, total_counts, on=category)
    merged_counts['proportion'] = merged_counts['count'] / merged_counts['total_count']

    if len(merged_counts) > 10:
        merged_counts = merged_counts.iloc[:10]
        
    with col1:
        fig_1 = px.bar(merged_counts, x=category, y='count', title='Value Counts of ' + category + " for the " + str(k) + " closest neighbors")
        st.plotly_chart(fig_1, use_container_width=True)
        
    with col2:
        fig_2 = px.pie(merged_counts.iloc[:5], values='proportion', names=category, title='Category Proportions')
        st.plotly_chart(fig_2, use_container_width=True)

