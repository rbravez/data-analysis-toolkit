import pandas as pd
import scipy.stats as st
from sklearn.covariance import EmpiricalCovariance
import numpy as np
import seaborn as sns
import sklearn.impute
import pandas as pd
import numpy as np
import statsmodels.api as sm
#from statsmodels.imputation.mice import MCAR
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

## Data Cleaning Tools

##

def exploratory_descriptive_analysis(df): 
    """
    Computes summary statistics, variance, skewness, and kurtosis for numerical columns.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: Summary statistics including variance, skewness, and kurtosis.
    """
    df_numeric = df.select_dtypes(include=[np.number])
    return (
        df_numeric.describe().T
        .assign(variance=df_numeric.var(), skewness=df_numeric.skew(), kurtosis=df_numeric.apply(st.kurtosis))
    )

def outlier_detector(df): ## Need to debug, isn't really realiable
    """
    Computes the outliers in the dataset via Z-score, Interquartile Range (IQR), 
    and Mahalanobis distance.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame (numerical columns only).
    
    Returns: 
        pd.DataFrame: Summary DataFrame marking outliers detected by each method.
        List: Indices of all detected outliers.
    """
    df_numeric = df.select_dtypes(include=[np.number])  # Select only numeric columns

    # Compute Z-score
    z_scores = st.zscore(df_numeric, nan_policy='omit')
    z_score_outliers = (np.abs(z_scores) > 3).any(axis=1)  # Mark rows with extreme values

    # Compute IQR
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).any(axis=1)

    # Compute Mahalanobis Distance
    cov = EmpiricalCovariance()
    cov.fit(df_numeric.dropna())  # Fit only on non-null values
    mahalanobis_dist = cov.mahalanobis(df_numeric.fillna(df_numeric.mean()))  # Replace NaNs with column means
    threshold = np.percentile(mahalanobis_dist, 97.5)  # Define an outlier threshold (adjustable)
    mahalanobis_outliers = mahalanobis_dist > threshold

    # Create output DataFrame
    summary_df = df_numeric.copy()
    summary_df["Z_Score_Outlier"] = z_score_outliers
    summary_df["IQR_Outlier"] = iqr_outliers
    summary_df["Mahalanobis_Distance"] = mahalanobis_dist
    summary_df["Mahalanobis_Outlier"] = mahalanobis_outliers

    # Get list of indices for detected outliers
    outlier_indices = summary_df[(summary_df["Z_Score_Outlier"]) | (summary_df["IQR_Outlier"]) | (summary_df["Mahalanobis_Outlier"])].index.tolist()

    return summary_df, outlier_indices

#def missing_data_analysis(df): --->>> Doesnt work
    """
    Analyzes missing data using MCAR and MAR techniques.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns: 
        pd.DataFrame: Summary of missing data count and percentage.
        dict: MCAR test results if applicable.
        Visualization: Heatmap of missing data.
    """

    # Count missing values
    missing_counts = df.isnull().sum()
    missing_percentage = (df.isnull().mean() * 100).round(2)

    # Create a DataFrame summary
    missing_summary = pd.DataFrame({
        "Missing Count": missing_counts,
        "Missing Percentage": missing_percentage
    }).sort_values(by="Missing Count", ascending=False)

    # Display missing value heatmap using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
    plt.title("Missing Data Heatmap")
    plt.show()

    # Doesnt work
    mcar_results = None
    try:
        mcar_test = MCAR(df.dropna(axis=1, how="all"))  # Drop columns that are entirely NaN
        mcar_results = mcar_test.fit()
    except Exception as e:
        print(f"MCAR Test failed: {e}")

    return missing_summary, mcar_results

def check_data_types(df):
    """
    Identifies mixed data types in columns and suggests corrections.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: Summary of data types and inconsistencies.
    """
    type_info = df.applymap(type).nunique()  # Count unique types per column
    inconsistent_cols = type_info[type_info > 1].index.tolist()  # Columns with mixed types
    return {"Inconsistent Columns": inconsistent_cols, "Data Types": df.dtypes}

def duplicate_data_analysis(df, drop=False):
    """
    Identifies and optionally removes duplicate rows.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        drop (bool): If True, drops duplicates and returns cleaned DataFrame.
    
    Returns:
        pd.DataFrame: Summary of duplicate counts.
        pd.DataFrame (optional): Cleaned DataFrame with duplicates removed.
    """
    duplicate_count = df.duplicated().sum()
    summary = pd.DataFrame({"Duplicate Rows": [duplicate_count]})
    
    if drop:
        return summary, df.drop_duplicates()
    return summary

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_features(df, method="standard"):
    """
    Scales numerical features using Standardization (Z-score) or Min-Max scaling.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        method (str): "standard" for StandardScaler, "minmax" for MinMaxScaler.
    
    Returns:
        pd.DataFrame: Scaled DataFrame.
    """
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df_numeric = df.select_dtypes(include=[np.number])  # Select numeric columns
    scaled_data = scaler.fit_transform(df_numeric)
    return pd.DataFrame(scaled_data, columns=df_numeric.columns)



def encode_categorical(df, method="onehot"):
    """
    Encodes categorical variables using one-hot encoding or label encoding.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        method (str): "onehot" for OneHotEncoder, "label" for LabelEncoder.
    
    Returns:
        pd.DataFrame: Encoded DataFrame.
    """
    df_cat = df.select_dtypes(include=['object', 'category'])  # Select categorical columns

    if method == "onehot":
        return pd.get_dummies(df, columns=df_cat.columns)
    else:
        for col in df_cat.columns:
            df[col] = LabelEncoder().fit_transform(df[col])
        return df
