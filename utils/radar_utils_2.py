from scipy.spatial.distance import euclidean, cosine
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle
import logging
import streamlit as st

MINERALS = ['B', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 
            'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'As', 'Br', 'Rb', 'Sr', 
            'Y', 'Zr', 'Nb', 'Cd', 'Sn', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 
            'Sm', 'W', 'Tl', 'Pb', 'U']
# Load bins info
bins_info_path = './data/bins_info.pkl'
with open(bins_info_path, 'rb') as f:
    bins_info = pickle.load(f)
# Function to plot radar chart
def plot_radar_chart(df, minerals, target=None, color=None):
    fig = go.Figure()
    for i in range(len(df)):
        values = df.iloc[i, :][minerals].values.flatten().tolist()
        values += values[:1]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=minerals + [minerals[0]],
            fill='toself',
            fillcolor=color,
            line_color=color,
            opacity=0.4 if target else 0.8,
            name=f'{target if target else "Input"}'
        ))

    title = "Radar Chart of Input Vector" if target is None else f"Group: {target}"
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        title={
            'text': title,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=370 if target else 550,
        width=370 if target else 550,
    )
    return fig

# Function to normalize column names
def normalize_column_names(columns):
    return [col.replace('(ppb)', '').strip() for col in columns]

def validate_columns(columns, required_columns, min_count):
    normalized_cols = normalize_column_names(columns)
    normalized_req = normalize_column_names(required_columns)
    return len(set(normalized_cols).intersection(set(normalized_req))) >= min_count

def rename_columns(df, required_columns):
    col_map = {col.replace('(ppb)', '').strip(): col for col in df.columns}
    normalized_required = normalize_column_names(required_columns)
    new_cols = {col_map[col]: col for col in normalized_required if col in col_map}
    return df.rename(columns=new_cols)

def replace_mineral_values(df, minerals):
    for mineral in minerals:
        df[mineral] = df[mineral].replace(['< LOD', ''], 0.00001).fillna(0.00001).astype(float)
    return df

def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

def calculate_feature_importance(data, features, target):
    X = data[features]
    y = data[target]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    importances = clf.feature_importances_
    return pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

def get_top_features(feature_importance_df, threshold=0.67):
    sorted_features = feature_importance_df.sort_values(by='Importance', ascending=False)
    return sorted_features.head(int(len(sorted_features) * threshold))['Feature'].tolist()

def downsample_balanced(df, target, n_samples_per_class):
    # Only include classes that have enough samples
    valid_classes = df[target].value_counts()[df[target].value_counts() >= n_samples_per_class].index
    df_balanced = pd.concat([
        resample(df[df[target] == cls],
                 replace=False,  # sample without replacement
                 n_samples=n_samples_per_class,  # to match n_samples_per_class
                 random_state=42)  # reproducible results
        for cls in valid_classes
    ])
    return df_balanced

def calculate_distances(group, new_bottle, features):
    return [euclidean(group.iloc[i][features], new_bottle[features].iloc[0]) for i in range(len(group))]

def is_within_range(new_bottle, group_stats, features, column):
    for feature in features:
        min_val = group_stats[(feature, 'min')] / 3
        max_val = group_stats[(feature, 'max')] * 3
        if not (min_val <= new_bottle[feature].iloc[0] <= max_val):
            return False
    return True

def calculate_top_k_distances(group, new_bottle, features, column):
    # Adjust group size based on its original size
    if len(group) > 5000:
        # If the group size is greater than 5000, reduce the size to a third
        indices = np.random.permutation(len(group))
        group = group.iloc[indices[:len(group)//4]]
    elif len(group) > 1000:
        # If the group size is greater than 100 but less than or equal to 5000, reduce the size to half
        indices = np.random.permutation(len(group))
        group = group.iloc[indices[:len(group)//2]]

    # Calculate k based on the possibly adjusted group size
    k = min(len(group) // 100, 8)
    k = max(k, 2)  # Ensure at least two distances are considered
    if column == 'Appellation':
        k = min(len(group) // 100, 8)
        k = max(k, 1)  # Ensure at least two distances are considered
    if column == 'Domaine':
 
        k = min(len(group) // 100, 8)
        k = max(k, 1)  # Ensure at least two distances are considered

    # Calculate distances for the potentially reduced group
    distances = [euclidean(group.iloc[i][features], new_bottle[features].iloc[0]) for i in range(len(group))]
    distances.sort()  # Sort distances to find the top k smallest values

    return distances[:k]  # Return the top k smallest distances

def find_top_k_groups(data, new_bottle, features, column, margin=2, predictions_dict=None):
    data = data.copy()
    if predictions_dict is not None:
        for key, value in predictions_dict.items():
            data = data[data[key] == value]

    if len(data) <= 1:
        st.error("Not enough data to compute similar groups, nor to train a model. Consider using regular predictions mode.")
        return None, None

    group_stats = data.groupby(column).agg({m: ['min', 'max'] for m in features})
    valid_groups = [group for group in group_stats.index if is_within_range(new_bottle, group_stats.loc[group], features, column)]

    group_distances = data[data[column].isin(valid_groups)].groupby(column).apply(lambda group: calculate_top_k_distances(group, new_bottle, features, column))
    average_distances = group_distances.apply(lambda dists: np.mean(dists) if len(dists) > 0 else float('inf')).reset_index()
    average_distances.columns = [column, 'Average Distance']

    average_distances = average_distances.sort_values(by='Average Distance')

    k_min = average_distances['Average Distance'].min()
    threshold = k_min + margin
    if column == 'Appellation':
        threshold = k_min + 3 * margin
    if column == 'Domaine':
        threshold = k_min + 3 * margin
    selected_groups = average_distances[average_distances['Average Distance'] < threshold]

    return selected_groups.iloc[:, 0].values, average_distances

def train_classifier(data, features, target, valid_groups, distances, predictions_dict=None, max_valid_groups=100):
    if target in ['Appellation', 'Domaine']:
        return None, None, None  # Skip training for 'Appellation' and 'Domaine'
    
    if predictions_dict:
        for key, value in predictions_dict.items():
            data = data[data[key] == value]

    train_df = data.dropna(subset=[target]).copy()
    train_df[target] = train_df[target].astype(str)
    
    if valid_groups is not None and len(valid_groups) > 0:
        valid_groups = [str(group) for group in valid_groups]
        
        if len(valid_groups) > max_valid_groups:
            valid_groups = np.random.choice(valid_groups, max_valid_groups, replace=False)
        
        train_df = train_df[train_df[target].isin(valid_groups)]

    unique_classes = train_df[target].unique()
    if len(unique_classes) < 2:
        return None, None, None
    
    class_mapping = {class_label: idx for idx, class_label in enumerate(unique_classes, start=0)}
    train_df[target] = train_df[target].map(class_mapping)

    valid_categories = train_df[target].value_counts()[train_df[target].value_counts() >= 2].index
    train_df = train_df[train_df[target].isin(valid_categories)]
    
    if train_df.empty:
        return None, None, None

    X = train_df[features]
    y = train_df[target]

    if len(y.unique()) < 2 or X.shape[0] <= 1:
        return None, None, None

    if distances is not None:
        weights = np.zeros_like(y, dtype=float)
        for group, dist in distances.itertuples(index=False):
            if group in class_mapping:
                mask = (train_df[target] == class_mapping[group])
                weights[mask] = (1 / (dist + 1e-6)) ** 3
        weights /= weights.mean()
        weights = weights * 2
    else:
        weights = np.ones_like(y, dtype=float)

    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42, stratify=y)

    if len(y_train.unique()) < 2 or X_train.shape[0] < 20:
        return None, None, None

    model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
    model.fit(X_train, y_train, svc__sample_weight=weights_train)
    class_mapping = {v: k for k, v in class_mapping.items()}

    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    renamed_report = {class_mapping[int(k)] if k.isdigit() else k: v for k, v in report.items()}
    
    return model, renamed_report, class_mapping



def display_prediction_results(model, input_vector, available_minerals, classes, training_data):
    probabilities = model.predict_proba(input_vector[available_minerals].values)
    max_prob_index = probabilities.argmax(axis=1)[0]
    prediction = model.predict(input_vector[available_minerals].values)[0]

    predicted_class = classes[prediction]
    probability = probabilities[0][max_prob_index]

    probas_on_training = model.predict_proba(training_data[available_minerals].values)
    class_indices = {v: k for k, v in classes.items()}

    mean_probas = {class_name: np.mean(probas_on_training[:, class_indices[class_name]]) for class_name in classes.values()}
    std_probas = {class_name: np.std(probas_on_training[:, class_indices[class_name]]) for class_name in classes.values()}

    mean_proba = mean_probas[predicted_class]
    std_proba = std_probas[predicted_class]

    st.markdown(f"### The predicted class is **{predicted_class}** with a probability of **{probability:.2f}**")

    if probability > mean_proba + std_proba:
        reliability = "Very High"
    elif probability > mean_proba:
        reliability = "High"
    elif probability > mean_proba - std_proba:
        reliability = "Medium"
    else:
        reliability = "Low"
    
    st.write(f"Reliability of the prediction: ***{reliability}***")
    #st.write(f"Mean probability for this class: ***{mean_proba:.2f}***")
    #st.write(f"Standard deviation of probabilities for this class: ***{std_proba:.2f}***")

    return predicted_class, probability, reliability


def reduce_pca(data, features, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data[features])
    reduced_data = pca.transform(data[features])
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    data[pca_columns] = pd.DataFrame(reduced_data, columns=pca_columns, index=data.index)
    return data, pca_columns, pca

def prepare_data_for_app2(df, minerals, min_vals, max_vals):
    available_minerals = [mineral for mineral in minerals if mineral in df.columns]
    df = df.copy()

    missing_minerals = [mineral for mineral in minerals if mineral not in df.columns]
    for mineral in missing_minerals:
        df[mineral] = 0.00001

    df = df[minerals]

    for mineral in minerals:
        min_val = min_vals[MINERALS.index(mineral)]
        max_val = max_vals[MINERALS.index(mineral)]
        df[mineral] = np.clip(df[mineral], min_val, max_val)

    df[minerals] = loaded_scaler.transform(df[minerals])
    return df

def prepare_data_for_app(df, bins_info, minerals=MINERALS):
    available_minerals = [mineral for mineral in minerals if mineral in df.columns]
    mineral_data = df[available_minerals].copy()
    
    mineral_data = mineral_data.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
    mineral_data = mineral_data.replace('< LOD', 0.00001).fillna(0.00001).astype(float)

    for col in available_minerals:
        lower_percentile = mineral_data[col].quantile(0.0001)
        upper_percentile = mineral_data[col].quantile(0.9999)
        mineral_data[col] = mineral_data[col].clip(lower=lower_percentile, upper=upper_percentile)

    for mineral in available_minerals:
        bins = bins_info[mineral]
        bin_indices = np.digitize(mineral_data[mineral], bins, right=True)
        
        bin_indices[bin_indices == 0] = 1
        bin_indices[bin_indices > 10] = 10
        
        mineral_data[mineral] = bin_indices

    df[available_minerals] = mineral_data
    return df

def plot_top_candidates(top_k_groups, average_distances, target, features, input_vector, normalized_data, MINERALS):
    columns = st.columns(min(len(top_k_groups), 4))  # Display a maximum of 4 columns
    colors = plotly.colors.qualitative.Plotly

    sorted_groups = average_distances.set_index(target).loc[top_k_groups].sort_values(by='Average Distance')

    for col_index, (target_group, row) in enumerate(sorted_groups.iterrows()):
        if col_index >= 4:
            break
        group_data = normalized_data[normalized_data[target] == target_group]
        representative_vector = group_data[features].median()

        combined_data = pd.concat([pd.DataFrame([representative_vector], columns=features), input_vector], axis=0)
        color = colors[col_index % len(colors)]

        fig = plot_radar_chart(combined_data, MINERALS, target_group, color)
        fig.add_trace(go.Scatterpolar(
            r=input_vector[MINERALS].values.flatten().tolist() + [input_vector[MINERALS].values.flatten().tolist()[0]],
            theta=MINERALS + [MINERALS[0]],
            fill='none',
            line_color='gray',
            name='Input Sample',
            line=dict(width=1)
        ))
        with columns[col_index]:
            st.plotly_chart(fig, use_container_width=True)

    if len(sorted_groups) > 4:
        additional_groups = sorted_groups.index[4:]
        selected_group = st.selectbox("See more groups", options=additional_groups)





