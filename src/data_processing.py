import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import itertools


# ignore warnings
import warnings
warnings.filterwarnings('ignore')



 
# # DATA CLEANING:



def df_data_cleaning(input_df):
    '''
    Function which takes in a given dataframe and removes duplicates, NA values, etc
    Also removes tracks which are muti-listed in multiple genres
    '''
    assert len(input_df) > 0 
    
    df = input_df
    initial_rows = df.shape[0]
    print(f"Initial number of rows: {initial_rows}")
    print(df.columns)

    # Check for invalid values and drop
    print(f"Number of invalid values {df.isna().sum()}")
    df = df.dropna()

    # Confirm drops successful
    after_null_drops_rows = df.shape[0]
    rows_dropped_from_null_drop = initial_rows-after_null_drops_rows
    print(f"Number of rows dropped from removing null, NA, etc values: {rows_dropped_from_null_drop}")
    print(f"Number of NA values after drops: {df.isna().sum()}")


    #Check for any duplicates
    # df.duplicated().sum()
    genre_mapping = df.groupby('track_id')['track_genre'].apply(lambda x: ', '.join(x)).reset_index()
    df_unique = df.drop_duplicates(subset=['track_id'], keep='first')
    df_unique = df_unique.drop(columns=['track_genre'])
    df = df_unique.merge(genre_mapping, on='track_id')
    after_duplicate_drop_rows = df.shape[0]
    rows_dropped_from_duplicate_drop = after_null_drops_rows- after_duplicate_drop_rows
    print(f"Number of rows dropped from removing duplicates: {rows_dropped_from_duplicate_drop}")

    return df





def describe_df_outliers(input_df):
    '''
    Function to take input datafram and describe outliers we might see within the data  in some categories (examples: tempo, loudness)
    '''
    df = input_df
    

    # Separate numerical features and categorical features
    num_cols = df.select_dtypes('number').columns
    # remove the unwanted auto-index column 'Unnamed: 0'
    num_cols = [col for col in num_cols if col.lower() != 'unnamed: 0']
    print(f"Remaining columns after removal of auto index: {num_cols}")
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    print(f"Categorical columns: {cat_cols}")


    # check unique values of bool-type feature "explicit" (should only be True and False)
    df['explicit'].value_counts().plot(kind='bar', title='Distribution of explicit')
    plt.ylabel('Count')
    plt.show()
    print("Number of explicit values: {df['explicit'].unique()}")


    # count of "outlier" points for each feature and their percentile
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = n_outliers / len(df) * 100
        print(f'{col:20s}  outliers: {n_outliers:6d}  ({pct:.2f}%)')

    
    # Datapoint with low tempo (-> 0) might be treated as abnormal -> no stable beats: ambient sound, white noise
    tempo_outliers = df[df['tempo'] == 0]
    print(f"Tempo ≈ 0 outliers: {len(tempo_outliers)} rows")

    # Datapoint with small loudness (smaller than a threshold, say 45) might be treated as abnormal -> Extremely quiet: likely silence, recording artifacts 
    loudness_outliers = df[df['loudness'] < -45]
    print(f"Loudness < -45 outliers: {len(loudness_outliers)} rows")