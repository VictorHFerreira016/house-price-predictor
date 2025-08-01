import pandas as pd

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    """Handles missing values in the DataFrame by filling them with appropriate values.
    Args: 
        df (pd.DataFrame): The DataFrame to process.
        Returns:
        pd.DataFrame: The DataFrame with missing values handled."""
    
    none_categories = ["PoolQC", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageType", 
                   "GarageFinish", "GarageCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                   "BsmtQual", "BsmtCond", "MasVnrType", "MiscFeature"]
    
    # Fill the NaN values with None
    for i in none_categories:
        df[i] = df[i].fillna("None")

    # Fill the "Electrical" feature with the mode
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    # Using "transform()" to fill the NaN values with the median, grouping each
    # LotFrontage according to the Neighborhood.
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median()))
        
    # Filling the NaN values with 0.
    if "MasVnrArea" in df.columns:
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

    return df

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies feature engineering to the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to process.
    Returns:
        pd.DataFrame: The DataFrame with engineered features."""
    
    # Here I added the Features related to the size of the house, like 
    # basement, 1st Floor and so on.
    total_area = ['1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFinSF1', 'BsmtFinSF2']
    
    # all() returns True, if all columns in the total_area are present in df.columns
    # 
    if all(col in df.columns for col in total_area):
        df['TotalLivingArea'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['LowQualFinSF'] + \
                                df['GrLivArea'] + df['BsmtFinSF1'] + df['BsmtFinSF2']
        
    else: 
        print("Not all required columns for TotalLivingArea are present in the DataFrame.")
        df['TotalLivingArea'] = 0
    return df

def preprocess_for_model(df: pd.DataFrame, trained_columns: list = None) -> pd.DataFrame:
    """Uses the DataFrame to preprocess all the data necessary to the model
    Args:
        df (pd.DataFrame): it is the DataFrame.
        trained_columns (list=None): are the columns already trained"""
    # It copies the DataFrame to do preprocessing
    df = handle_missing_values(df.copy())
    df = apply_feature_engineering(df.copy())

    # Creates numeric columns (int, float)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Colunas categóricas
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # It removes the Id column, because it is not necessary to this model
    if 'Id' in numeric_cols:
        numeric_cols.remove('Id')
    if 'SalePrice' in numeric_cols:
        numeric_cols.remove('SalePrice')

    # It transform categorical features in binary features
    # Drop_first is used to evitate multicollinearity
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # If trained columns is provided
    if trained_columns is not None:
        # Identifies missing columns comparing with the ones used on training
        missing_cols = set(trained_columns) - set(df_processed.columns)

        for i in missing_cols:
            df_processed[i] = 0
        
        df_processed = df_processed[trained_columns]

    # Fill in any remaining missing values with 0.
    df_processed = df_processed.fillna(0)

    return df_processed

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Load the data specified and process it."""
    df = pd.read_csv(file_path)
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    df = handle_missing_values(df.copy())
    return df