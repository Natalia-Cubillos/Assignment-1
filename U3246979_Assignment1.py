#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: Natalia Andrea Cubillos Villegas
Student ID: U3246979 

Data Science Technology and Systems
Assignment 1: Predictive Modelling of Eating-Out Problem
"""

# ***** Loading required packages and libraries *****
import ast, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.impute import KNNImputer
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline



# //////////////////////////////////////////////////////////////////////////

''' Part A: Exploratory Data Analysis '''

# ***** Load and Explore the data *****
dataset = pd.read_csv('zomato_df_final_data.csv') # Loading the data
# Checking the first five rows
print(dataset.head()) 
# Checking missing values per column
print('Missing Values per Column\n', dataset.isnull().sum())
# Total number of rows and columns
print('\nTotal number of rows and columns:\n', dataset.shape)
# Column types
print('\nColumn Types:\n', dataset.dtypes)
# Summary statistics for numerical columns
print('\nSummary Statistics:\n', dataset.describe())

# ***** Plotting the Missing Values Percentage *****
missing_values_per = (dataset.isnull().sum() / len(dataset))*100
ax = missing_values_per.plot(kind = 'bar', figsize = (10,6))
plt.title('Missing Values Percentage per column')
plt.ylabel('Percentage (%)')
plt.xlabel('Feature')
plt.ylim(0,40)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%',
                (p.get_x()+p.get_width()/2., p.get_height()),
                ha = 'center', va = 'bottom', fontsize = 9, color = 'black',
                rotation = 45, xytext = (0,3), textcoords = 'offset points')
plt.show()

# ////////////////////////////////////////////////////////////////////////////

''' Moderate rate of missing values is found in the features Rating Number,
    Rating Text and Votes, with 32% of missing data approximately. A low rate 
    of missing values is found for features such as cost, lat, lng, type and 
    cost_2.
    
    For the features, cost and cost_2, the median will be imputed in the
    missing values. For the features lat (latituted), lng (longitud), and type,
    the missing values will be removed as the percentage is not significant.
    Additionally, for the Rating number and Votes, a KNNImputer will be 
    processed. Finally, Rating Text will be imputed according to its 
    correlation with Rating Number'''
    

# ***** Missing Values Imputation *****

# Imputing Median in cost and cost_2
for i in ['cost', 'cost_2']:
    if i in dataset.columns:
        med = dataset[i].median()
        dataset[i] = dataset[i].fillna(med)

# Imputation using KNNImputer for Ratings Number & Votes
targets = [c for c in ["rating_number", "votes"] if c in dataset.columns]
if targets:
    predictors = [c for c in ["cost", "cost_2", "lat", "lng"] if c in dataset.columns]
    cols = [*targets, *predictors]
    cols = list(dict.fromkeys(cols))

    if cols:
        # KNN imputation
        work = dataset[cols].copy()
        # To avoid changing non-missing-values
        nan_mask = work[targets].isna()
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        imputed = pd.DataFrame(imputer.fit_transform(work), 
                               columns=cols, index=dataset.index)

        # Imputing ONLY missing values
        for t in targets:
            dataset.loc[nan_mask[t], t] = imputed.loc[nan_mask[t], t]
else:
    pass

# Dropping the rows with missing lat/lng as they are crucial for mapping
must_have = [c for c in ['lat', 'lng'] if c in dataset.columns]
if must_have:
    dataset = dataset.dropna(subset=must_have)

# Handling missing 'type' by imputing with the mode 
if 'type' in dataset.columns:
    mode_type = dataset['type'].mode(dropna=True)
    if not mode_type.empty:
        dataset['type'] = dataset['type'].fillna(mode_type[0])

    
# Checking correlation between Rating Number and Rating Text
print(dataset.groupby('rating_text')['rating_number'].mean())

# Replacing Missing Values in Rating Text as per correlation
def map_rating(x):
    if pd.isna(x):
        return np.nan
    elif x >= 4.5:
        return 'Excellent'
    elif x >= 4.0:
        return 'Very Good'
    elif x >= 3.5:
        return 'Good'
    elif x >= 2.5:
        return 'Average'
    else:
        return 'Poor'

dataset['rating_text'] = dataset['rating_number'].apply(map_rating)
print(dataset.groupby("rating_text")["rating_number"].mean())

#/////////////////////////////////////////////////////////////////////////////

# ***** How many unique cuisines are served? *****
# Converting strings into lists
df_cuisine = dataset['cuisine'].apply(ast.literal_eval)
# Flattening list of cuisines into one list to check frequencies
list_cuisines = [c for sublist in df_cuisine for c in sublist]

# Converting to Pandas Series for Counting 
counts_cuisines = pd.Series(list_cuisines).value_counts()
print('Number of Unique Cuisines:', counts_cuisines.nunique())
print('\nTop 10 cuisines:', counts_cuisines.head(10).index.tolist())

# Plotting the Top 10 cuisines
ax = counts_cuisines.head(10).plot(kind='bar', color='skyblue', figsize=(10,6))
plt.ylim(0,3500)
plt.title('Top 10 Cuisines')
plt.ylabel('Number of Restaurants')
plt.xlabel('Cuisine')
plt.xticks(rotation=45, ha='right')
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=9, rotation=45,
                xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.show()

#////////////////////////////////////////////////////////////////////////////

# ***** Which 3 suburbs have the most restaurants? *****
# Selecting the first three suburbs as per the count
top_suburbs = dataset["subzone"].value_counts().head(3)
print('Top 3 suburbs with most restaurants:', top_suburbs)

# Plotting the top 5 suburbs
ax = dataset["subzone"].value_counts().head(5).plot(kind='bar', color='pink', 
                                                    figsize=(10,6))
plt.ylim(0,500)
plt.title('Top 5 Suburbs with most restaurants')
plt.ylabel('Number of Restaurants')
plt.xlabel('Suburb')
plt.xticks(rotation=45, ha='right')
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=9, rotation=45,
                xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.show()
    
#////////////////////////////////////////////////////////////////////////////
# Are restaurants with “Excellent” ratings more expensive than those with 
# “Poor” ratings?

# Showing the statistics of the cost by rating
cost_by_rating = dataset.groupby('rating_text')['cost'].describe()
    
# Filtering Excellent vs Poor
exc_vs_poor = dataset[dataset['rating_text'].isin(['Excellent', 'Poor'])]

# Plotting Histigram (frequency of the resturants within a category and cost)
plt.figure(figsize=(10,6))
for label, color in zip(['Excellent', 'Poor'], ['orange','blue']):
    exc_vs_poor[exc_vs_poor['rating_text'] == label]['cost'].plot(
        kind='hist', alpha=0.5, bins=20, color=color, label=label
    )

plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.title('Cost by Rating (Excellent vs Poor)')
plt.legend()
plt.show()
    
# Plotting Stocked Bars for comparison
dataset['cost_bin'] = pd.cut(dataset['cost'], bins=[0,20,50,100,200,500], 
                             right=False)

cost_bins = pd.crosstab(dataset['cost_bin'], dataset['rating_text']
                 [dataset['rating_text'].isin(['Excellent', 'Poor'])])

cost_bins.plot(kind='bar', stacked=True, figsize=(10,6), color=['pink','blue'])
plt.title('Cost Bins vs Rating (Excellent vs Poor)')
plt.ylabel('Number of Restaurant')
plt.show()
    
#////////////////////////////////////////////////////////////////////////////
# Plotting the Distribution of Cost
plt.figure(figsize=(8,5))
sns.histplot(dataset['cost'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Restaurants Cost')
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.show()
    
# Plotting the Distribution of Ratings
plt.figure(figsize=(8,5))
sns.countplot(x='rating_text', data=dataset, 
              order=['Poor','Average','Good','Very Good','Excellent'])
plt.title('Distribution of Restaurants Ratings')
plt.xlabel('Rating Category')
plt.ylabel('Count')
plt.show()

# Plotting the Dristibution of Restaurant Types
df_type = dataset['type'].apply(ast.literal_eval)
list_types = [t for sublist in df_type for t in sublist]

counts_types = pd.Series(list_types).value_counts()
print('Number of Unique Types:', counts_types.nunique())
print('\nTop 3 types:', counts_types.head(3).index.tolist())

# Plot Top 10 restaurant types
ax = counts_types.plot(kind='bar', color='orange', figsize=(10,6))
plt.title('Distribution of Restaurant Types')
plt.ylabel('Number of Restaurants')
plt.xlabel('Type')
plt.ylim(0,6000)
plt.xticks(rotation=45, ha='right')
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=9, rotation=45,
                xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.show()
    
#////////////////////////////////////////////////////////////////////////////
# Correlation between Cost and Votes
sns.scatterplot(x='cost', y='votes', data=dataset, alpha=0.5)
plt.title('Correlation Between Cost and Votes')
plt.xlabel('Cost')
plt.ylabel('Votes')
plt.show()
print('Correlation:', dataset['cost'].corr(dataset['votes']))

# Correlation between Cost and Ratings
sns.scatterplot(x='cost', y='rating_number', data=dataset, alpha=0.5)
plt.title('Correlation Between Cost and Rating')
plt.xlabel('Cost')
plt.ylabel('Rating Number')
plt.show()
print('Correlation:', dataset['cost'].corr(dataset['rating_number']))

# //////////////////////////////////////////////////////////////////////////
# Geospatial Analysis using Geopandas to show cuisine density per suburb

# Loading Sydney suburb boundaries
geo_syd = gpd.read_file('sydney.geojson')

# Ensuring WGS84 for Plotly, required for Interactive Plot
if geo_syd.crs is None or geo_syd.crs.to_epsg() != 4326:
    geo_syd = geo_syd.to_crs(4326)

# Reading Suburbs Names
geo_syd_key = 'SSC_NAME'
# Creating a column to use this format for joins
geo_syd['ssc_name_lc'] = geo_syd[geo_syd_key].astype(str).str.strip().str.lower()
# Changing Suburb Names to match the geojson key (original dataset)
dataset['subzone'] = dataset['subzone'].astype(str).str.strip().str.lower()

# Preparing restaurant data as cuisine types are strings
if dataset['cuisine'].dtype == 'object' and dataset['cuisine'].astype(str).str.startswith('[').any():
    # Converting strings into a Python list
    dataset['cuisine'] = dataset['cuisine'].apply(ast.literal_eval)

# Splitting cuisines to one row per restaurant-cuisine and respective suburb
df_cuisine_sub = dataset[['subzone', 'cuisine']].explode('cuisine').dropna()
df_cuisine_sub['cuisine'] = df_cuisine_sub['cuisine'].astype(str).str.strip()

# Defining function to build counts per suburb for a cuisine
def cuisine_counts_for(cuisine_name: str) -> gpd.GeoDataFrame:
    mask = df_cuisine_sub['cuisine'].str.casefold() == cuisine_name.casefold()
    counts = (df_cuisine_sub.loc[mask]
              .groupby('subzone').size()
              .reset_index(name='count'))

    # join lowercase→lowercase
    merged_geo_data = geo_syd.merge(counts, left_on='ssc_name_lc', 
                                    right_on='subzone', how='left')
    merged_geo_data['count'] = merged_geo_data['count'].fillna(0).astype(int)

    # ensure SSC_NAME is trimmed (no stray spaces)
    merged_geo_data['SSC_NAME'] = merged_geo_data['SSC_NAME'].astype(str).str.strip()
    return merged_geo_data

# Defining function to plot suburbs and cuisine density
def plot_cuisine_counts(cuisine_name: str):
    merged_geo_data = cuisine_counts_for(cuisine_name)
    ax = merged_geo_data.plot(
        column='count',
        cmap='OrRd',
        linewidth=0.3,
        edgecolor='white',
        legend=True,
        legend_kwds={'label': f'Number of {cuisine_name} restaurants',
                     'orientation': 'vertical'},
        missing_kwds={'color': '#f0f0f0', 'hatch': '///', 'label': 'No data'},
        figsize=(9, 7))
    ax.set_title(f'{cuisine_name} cuisine by suburb', pad=12)
    ax.axis('off')
    plt.show()

# Example:
plot_cuisine_counts('Japanese')
plot_cuisine_counts('Thai')

# //////////////////////////////////////////////////////////////////////////
# Interactive Geospatial Analysis using Plotly

# Opening the map in browser for better visualisation
pio.renderers.default = 'browser'

# Creating a list of Cuisine Names for dropdown list
cuisine_list = ['Japanese', 'Cafe', 'Thai', 'Chinese', 'Italian', 
                'Modern Australian', 'Burger', 'Indian']

# Using the SAME GeoJSON 
geojson_full = json.loads(geo_syd.to_json())

fig = go.Figure()

# Fixing color scale across cuisines for fair comparison
global_max = 1
buffers = []
for name in cuisine_list:
    mgd = cuisine_counts_for(name)
    dfp = mgd[['SSC_NAME', 'count']].copy()
    buffers.append(dfp)
    global_max = max(global_max, int(dfp['count'].max()))

for i, (name, dfp) in enumerate(zip(cuisine_list, buffers)):
    fig.add_choroplethmapbox(
        geojson=geojson_full,
        locations=dfp['SSC_NAME'],
        featureidkey='properties.SSC_NAME',
        z=dfp['count'],
        colorscale='OrRd',
        zmin=0, zmax=global_max,
        marker_line_width=0.2, marker_line_color='white',
        visible=(i == 0),
        name=name,
        hovertext=dfp['SSC_NAME'] + ': ' + dfp['count'].astype(str),
        hoverinfo='text',
        showscale=True)

buttons = []
for i, name in enumerate(cuisine_list):
    vis = [False]*len(cuisine_list); vis[i] = True
    buttons.append(dict(label=name, method='update',
                        args=[{'visible': vis},
                              {'title': f'Interactive Map: {name} restaurants by suburb'}]))

# Computing Sydney bounds
minx, miny, maxx, maxy = geo_syd.total_bounds

fig.update_layout(
    mapbox=dict(
        style='carto-positron',
        center={'lat': -33.87, 'lon': 151.21},
        zoom=9,
        bounds={'west': float(minx), 'east': float(maxx),
                'south': float(miny), 'north': float(maxy)}),
    updatemenus=[dict(active=0, buttons=buttons, x=0.05, y=1.08, xanchor='left')],
    margin={'r':0,'t':40,'l':0,'b':0},
    title=f'Interactive Map: {cuisine_list[0]} restaurants by suburb')

fig.show()

# //////////////////////////////////////////////////////////////////////////
''' Part B – Predictive Modelling '''

# ***** Feature Engineering *****
# Encoding categorical features properly (Label Encoding, One-Hot, etc.)

''' Nominal categories: cuisine, subzone, type -- In this case, the
    best choice is One-Hot Encoding as it avoids introducing artificial 
    ranking which is not required because these features are unordered.
    Ordinal categories: rating_text -- In this case, the best choice
    is Label Encoding with order preserved (mapping), as this preserves 
    the ranking, which might be useful for models that use order such as 
    linear models and tree-based models. '''
    
df = dataset.copy()

# Defining function to convert text into lists
def to_list_if_str(x):
    if isinstance(x, str) and x.strip().startswith('['):
        try:
            return ast.literal_eval(x)
        except Exception:
            return [x]
    return x

# Applying Label Encoding for the categorical variable 'rating_text'
rating_order = ['Poor', 'Average', 'Good', 'Very Good', 'Excellent']
# Mapping numbers to words so the model can understand order
rating_map = {k: i+1 for i, k in enumerate(rating_order)}
df['rating_text_ord'] = df['rating_text'].map(rating_map)

# Applying One-Hot Encoding for the categorical variable 'Cuisine'
if 'cuisine' in df.columns:
    # Converting text into lists
    df['cuisine'] = df['cuisine'].apply(to_list_if_str)
    # Ensuring all rows are lists
    df['cuisine'] = df['cuisine'].apply(
        lambda v: v if isinstance(v, (list, tuple, set)) else [str(v)])
    # Cleaning spaces inside names
    df['cuisine'] = df['cuisine'].apply(
        lambda lst: [str(x).strip() for x in lst if str(x).strip() != ''])
    # Applyingy One-Hot Encoding creating new columns for each cuisine
    mlb_cuisine = MultiLabelBinarizer(sparse_output=False)
    cuisine_ohe = pd.DataFrame(
        mlb_cuisine.fit_transform(df['cuisine']),
        columns=[f'cuisine_{c}' for c in mlb_cuisine.classes_],
        index=df.index)
    # Joining new one-hot columns with original dataframe
    df = pd.concat([df, cuisine_ohe], axis=1)

# Applying One-Hot Encoding for the categorical variable 'Type'
if 'type' in df.columns:
    # Converting text to list
    df['type'] = df['type'].apply(to_list_if_str)
    # Ensuring every row is a list 
    df['type'] = df['type'].apply(
        lambda v: v if isinstance(v, (list, tuple, set)) else [str(v)])
    # Cleaning spaces inside each type value
    df['type'] = df['type'].apply(
        lambda lst: [str(x).strip() for x in lst if str(x).strip() != ''])
    # Applying one-hot encoding creating one column per type
    mlb_type = MultiLabelBinarizer(sparse_output=False)
    type_ohe = pd.DataFrame(
        mlb_type.fit_transform(df['type']),
        columns=[f'type_{t}' for t in mlb_type.classes_],
        index=df.index)
    # Joining new columns with original dataframe
    df = pd.concat([df, type_ohe], axis=1)

# Applying One-Hot Encoding for the categorical variable 'subzone'
# Counting the number of restaurants in each suburb
df['subzone_freq'] = df['subzone'].map(df['subzone'].value_counts()).astype(int)
# Creating one column per suburb 
subzone_ohe = pd.get_dummies(df['subzone'], prefix='subzone', drop_first=False)
# Joining the one-hot encoded suburbs back to the dataframe
df = pd.concat([df, subzone_ohe], axis=1)


# Dropping raw categorical columns after encoding
cols_to_drop = ['rating_text', 'cuisine', 'type', 'subzone']
df_model = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# //////////////////////////////////////////////////////////////////////////
# ***** Creating useful features *****

''' As part of the Feature Engineering process, four new features are created:
    1. Cuisine Diversity: This help us to identify how many cuisines a 
    restaurant serves.
    2. Cost Bins: This makes cost interpretable as it helps capture non-linear 
    relationships by assigning bins such as Low_Cost = [0–50], 
    Medium_Cost = [50–100], High_Cost = [100–200], Luxury = 200+.
    3. Votes intensity: It helps distinguish restaurants with high visibility
    by calculatiing the median and flagging those above this measure.
    4. Suburb Density: This variable keep the count of the restaurants per
    suburb, this helps identify high density areas which is relevant as this
    can change competition, prices, and ratings.
    '''

# Creating a new feature called 'Cuisine diversity'
cuisine_cols = [c for c in df_model.columns if c.startswith('cuisine_')]
df_model[cuisine_cols] = df_model[cuisine_cols].apply(
    pd.to_numeric, errors='coerce').fillna(0).astype(int)
df_model['cuisine_diversity'] = df_model[cuisine_cols].sum(axis=1).astype(int)

# Creating a new feature called 'Cost Bins'
bins = [0, 50, 100, 200, 500]
labels = ['Low', 'Medium', 'High', 'Luxury']
df_model['cost_bin'] = pd.cut(df_model['cost'], bins=bins, labels=labels, 
                              include_lowest=True)
# Creating a new feature called 'Votes intensity'
median_votes = df_model['votes'].median()
df_model['votes_intensity'] = (df_model['votes'] > median_votes).astype(int)

# Creating a new feature called 'Suburb Density'
# Building the vector of total counts per suburb from one-hot columns
subzone_onehot_cols = [c for c in df_model.columns if c.startswith('subzone_')]
subzone_totals = df_model[subzone_onehot_cols].sum(axis=0)
# Checking the total per suburb
df_model['suburb_density'] = (df_model[subzone_onehot_cols] @
                              subzone_totals).astype(int)


# Checking new data frame after encoding and adding features
print(df_model.head()) # Checking the first five rows
# Checking missing values per column
print('Missing Values per Column\n', df_model.isnull().sum())
# Total number of rows and columns
print('\nTotal number of rows and columns:\n', df_model.shape)
# Column types
print('\nColumn Types:\n', df_model.dtypes)
# Summary statistics for numerical columns
print('\nSummary Statistics:\n', df_model.describe())

    
df_model = df_model.copy()
print(df_model.info())

# //////////////////////////////////////////////////////////////////////////
# ***** Regression Models on Rating Number *****

# Linear Regression
# Defining Predictors & Target Variable
X = df_model.drop(columns=['rating_number'])
# Keeping only numerical variables
X = X.select_dtypes(include=[np.number])
# Target variable is 'rating_number'
y = df_model['rating_number']

# Splitting dataset into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

# Fitting Linear Regression Model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

# Predictions on the test set
y_pred_linear_regression = linear_regression_model.predict(X_test)

''' Definition: Mean Squared Error and R-squared are common regression model 
    accuracy metrics where a lower MSE indicates better accuracy, and a higher 
    R2 (closer to 1) also signifies a better model fit.'''

# Accuracy metrics
mse_linear_regression = mean_squared_error(y_test, y_pred_linear_regression)
r2_linear_regression = r2_score(y_test, y_pred_linear_regression)

# //////////////////////////////////////////////////////////////////////////
# Linear Regression with Gradient Descent Regression

# Scaling features
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled  = feature_scaler.transform(X_test)

# Ensuring float dtype for safety
X_train_scaled = X_train_scaled.astype(np.float64)
X_test_scaled  = X_test_scaled.astype(np.float64)
y_train_np = y_train.to_numpy(dtype=np.float64)
y_test_np  = y_test.to_numpy(dtype=np.float64)

# Defining the model
sgd_regression_model = SGDRegressor(
    loss='squared_error',
    penalty='l2',
    alpha=1e-4,   
    learning_rate='constant',
    eta0=1e-5,
    max_iter=20000,
    tol=1e-7,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    average=True,
    random_state=42)

# Fitting the model
sgd_regression_model.fit(X_train_scaled, y_train_np)

# Predictions on the test set
y_pred_sgd = sgd_regression_model.predict(X_test_scaled)

# Accuracy metrics
mse_gradient_descent = mean_squared_error(y_test_np, y_pred_sgd)
r2_gradient_descent = r2_score(y_test_np, y_pred_sgd)

# //////////////////////////////////////////////////////////////////////////
# Linear Regression with PCA 

''' An additional analysis was performed using Principal Component Analysis.
    The goal was to check if model stability improved after reducing the 
    number of features, since many dummy variables were created during 
    Feature Engineering. '''

# Defining Predictors & Target Variable
X_full_features = df_model.drop(columns=['rating_number'])
# Keeping only numerical variables
X_full_features = X_full_features.select_dtypes(include=[np.number])
y_target_rating_number = df_model['rating_number']

# Splitting dataset into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_full_features, y_target_rating_number, test_size=0.2, random_state=42)

# Standardising predictors 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Applying PCA keeping 95% variance
pca = PCA(n_components=0.95, svd_solver='full', random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print('# PCA kept components:', pca.n_components_)
print('# PCA cumulative variance explained:', pca.explained_variance_ratio_.sum())

# Fitting Linear Regression on PCA features
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train_pca, y_train)

# Predictions on the test set
y_pred_linear_regression = linear_regression_model.predict(X_test_pca)

# Accuracy metrics
mse_linear_regression_pca = mean_squared_error(y_test, y_pred_linear_regression)
r2_linear_regression_pca = r2_score(y_test, y_pred_linear_regression)

# //////////////////////////////////////////////////////////////////////////
# ***** Classification Models on Rating Text *****

# Creating binary target variable
rating_binary_map = {
    'Poor': 0, 'Average': 0,
    'Good': 1, 'Very Good': 1, 'Excellent': 1}

df_model['rating_binary'] = df['rating_text'].map(rating_binary_map)

# Defining predictors (X) and target variable (Y)
X = df_model.drop(columns=['rating_text_ord', 'rating_binary', 'rating_number'])
X = X.select_dtypes(include=[np.number])
y = df_model['rating_binary']

# Splitting dataset into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                random_state=42, stratify=y)

# Defining function to train and evaluate classifiers
def classification_models_evaluation(model, model_name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        'Model': model_name,
        'Confusion Matrix': cm,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1}

# Logistic Regression Model
log_reg_model = LogisticRegression(max_iter=500, solver='lbfgs')

# Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=400, max_depth=12,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1)

# Support Vector Machine
svm_model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))])

# Neural Network Model
nn_model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('clf', MLPClassifier(hidden_layer_sizes=(64, 64),
                          activation='relu',
                          alpha=1e-4,
                          max_iter=1000,
                          early_stopping=True,
                          random_state=42))])

# Evaluating all models
results = []
results.append(classification_models_evaluation(log_reg_model, 
                                            "Logistic Regression", 
                                            X_train, X_test, y_train, y_test))
results.append(classification_models_evaluation(rf_model, 
                                            "Random Forest", 
                                            X_train, X_test, y_train, y_test))
results.append(classification_models_evaluation(svm_model,
                                            "Support Vector Machine", 
                                            X_train, X_test, y_train, y_test))
results.append(classification_models_evaluation(nn_model,
                                            "Neural Network (MLP)", 
                                            X_train, X_test, y_train, y_test))

# //////////////////////////////////////////////////////////////////////////
# ***** Final Comparison: Printing Results in a table *****

print('\n================== Regression Models Comparison ====================')
print(f"Model A | Linear Regression without PCA | MSE = {mse_linear_regression:.4f} | R2 = {r2_linear_regression:.4f}")
print(f"Model B |  Gradient Descent Regression  | MSE = {mse_gradient_descent:.4f} | R2 = {r2_gradient_descent:.4f}")
print(f"Model C |   Linear Regression with PCA  | MSE = {mse_linear_regression_pca:.4f} | R2 = {r2_linear_regression_pca:.4f}")
print('====================================================================\n')

print('\n=============================== Classification Models Comparison =================================')
for r in results:
    print(f"{r['Model']:25} | "
          f"Accuracy = {r['Accuracy']:.4f} | "
          f"Precision = {r['Precision']:.4f} | "
          f"Recall = {r['Recall']:.4f} | "
          f"F1 = {r['F1 Score']:.4f}")
print('==================================================================================================\n')

print('\n===================== Confusion Matrix Comparison ======================')
for r in results:
    print(f"{r['Model']:25} | "
          f"Confusion Matrix = {r['Confusion Matrix'].tolist()}")
print('========================================================================\n')

# //////////////////////////////////////////////////////////////////////////
# ***** Regression and Classification using PySpark MLlib pipelines *****

# Loading Required Packages and Libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression as SparkLinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import time
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# PySpark setup 
spark = SparkSession.builder \
    .appName('uc-assignment-pyspark') \
    .getOrCreate()
    
# Keeping only numeric columns
numeric_df = df_model.select_dtypes(include=[np.number]).copy()
spark_df = spark.createDataFrame(numeric_df)

# //////////////////////////////////////////////////////////////////////////
# Linear Regression on Rating Number using PySpark

# Defining predictors and target variable
target_reg = 'rating_number'
feature_cols_reg = [c for c in spark_df.columns if c != target_reg]

# Assembling Features Vector
assembler_reg = VectorAssembler(inputCols=feature_cols_reg,
                                outputCol='features_raw')
scaler_reg    = StandardScaler(inputCol='features_raw',
                               outputCol='features',
                               withMean=True, withStd=True)

# Defining Linear Regression Model using PySpark
lr = SparkLinearRegression(featuresCol='features',
                                     labelCol=target_reg,
                                     predictionCol='prediction')

# Building the pipeline
pipe_reg = Pipeline(stages=[assembler_reg, scaler_reg, lr])

# Splitting into Test (20%) and Train (80%) sets
train_reg, test_reg = spark_df.randomSplit([0.8, 0.2], seed=42)

# Train timing
t0 = time.perf_counter()
model_reg = pipe_reg.fit(train_reg)
t_train = time.perf_counter() - t0

# Inference timing
t1 = time.perf_counter()
pred_reg = model_reg.transform(test_reg).cache()
_ = pred_reg.count()
t_infer = time.perf_counter() - t1

# Evaluating the model: Metrics (MSE & R2)
evaluator_mse = RegressionEvaluator(labelCol=target_reg,
                                    predictionCol='prediction',
                                    metricName='mse')
evaluator_r2 = RegressionEvaluator(labelCol=target_reg, 
                                   predictionCol='prediction', 
                                   metricName='r2')

mse_pyspark = evaluator_mse.evaluate(pred_reg)
r2_pyspark  = evaluator_r2.evaluate(pred_reg)

print('\n============= PySpark Linear Regression ==============')
print(f'           MSE = {mse_pyspark:.6f} | R2 = {r2_pyspark:.6f}')
print(f'   Train time (s) = {t_train:.3f} | Inference time (s) = {t_infer:.3f}')
print('======================================================\n')

# //////////////////////////////////////////////////////////////////////////
# Logistic Regression on Rating Text using PySpark

# Defining Predictors and Target Variable
target_clf = 'rating_binary'
feature_cols_clf = [c for c in spark_df.columns if c != target_clf and c != 'rating_number']

# Assembling features vector
assembler_clf = VectorAssembler(inputCols=feature_cols_clf, 
                                outputCol='features_raw')
scaler_clf    = StandardScaler(inputCol='features_raw', 
                               outputCol='features', 
                               withMean=True, withStd=True)

# Logistic Regression Model with Regularization L2
logreg = LogisticRegression(featuresCol='features', labelCol=target_clf, 
                            predictionCol='prediction', maxIter=100,
                            regParam=0.1, elasticNetParam=0.0)

# Building Pipeline
pipe_clf = Pipeline(stages=[assembler_clf, scaler_clf, logreg])

# Splitting into Train(80%) and Test (20%) sets
train_clf, test_clf = spark_df.randomSplit([0.8, 0.2], seed=42)

# Timing train
t0 = time.perf_counter()
model_clf = pipe_clf.fit(train_clf)
t_train_cls = time.perf_counter() - t0

# Timing inference
t1 = time.perf_counter()
pred_clf = model_clf.transform(test_clf).cache()
_ = pred_clf.count()
t_infer_cls = time.perf_counter() - t1

# Evaluating the model:  Metrics (Accuracy, Scalability, and Speed)
eval_acc = MulticlassClassificationEvaluator(
    labelCol=target_clf, predictionCol='prediction', metricName='accuracy'
)

acc = eval_acc.evaluate(pred_clf)

print('=========== PySpark Logistic Regression ===========')
print(f'                 Accuracy = {acc:.4f}')
print(f'Train time (s) = {t_train_cls:.3f} | Inference time (s) = {t_infer_cls:.3f}')
print('===================================================\n')

# //////////////////////////////////////////////////////////////////////////
# ***** Final Comparison: PySpark vs Scikit-Learn *****

# Extracting Logistic Regression accuracy from results
sklearn_log_reg_acc = [r['Accuracy'] for r in results if r['Model'] == "Logistic Regression"][0]

print('======== PySpark vs Scikit-Learn =========')
print('Regression Model: Linear Regression')
print(f'Scikit-Learn  | MSE = {mse_linear_regression:.4f} | R2 = {r2_linear_regression:.4f}')
print(f'PySpark       | MSE = {mse_pyspark:.4f} | R2 = {r2_pyspark:.4f}')
print('==========================================')
print('Classification Model: Logistic Regression')
print(f'Scikit-Learn  | Accuracy = {sklearn_log_reg_acc:.4f}')
print(f'PySpark       | Accuracy = {acc:.4f}')
print('==========================================\n')





























    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    