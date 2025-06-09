from json import load
import pandas as pd
import numpy as np
# load parquet file
def load_parquet(file_path: str) -> pd.DataFrame:
    """
    Load a parquet file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the parquet file.
    
    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return pd.DataFrame()

def preprocessing_volume_and_buy_qty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame by converting 'volume' and 'buy_qty' to numeric.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.
    
    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    df['buy_qty_diff'] = df['buy_qty'].diff().fillna(0)
    df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
    
    return df

def preprocessing_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the DataFrame by scaling 'volume' and 'buy_qty'.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to normalize.
    
    Returns:
    pd.DataFrame: The normalized DataFrame.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    known_cols = ['timestamp', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'label']
    anon_cols = [col for col in df.columns if col not in known_cols]
    df = df.clip(-1e6, 1e6)

    X_scaled = scaler.fit_transform(df[anon_cols])
    
    df_scaled = pd.DataFrame(X_scaled, columns=[anon_cols])
    # substituting the scaled values back into the original DataFrame
    df.update(df_scaled)
    df = df.reset_index()

    return df

# save new df to parquet file
def save_to_parquet(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the DataFrame to a parquet file.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    file_path (str): The path to save the parquet file.
    """
    try:
        df.to_parquet(file_path, index=False)
        print(f"DataFrame saved to {file_path}")
    except Exception as e:
        print(f"Error saving DataFrame to parquet: {e}")

def preprocessing_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'timestamp' column to datetime and set it as index.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.
    
    Returns:
    pd.DataFrame: The DataFrame with 'timestamp' as index.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek  # 0=Monday
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    return df


def preprocessing_and_save(df: pd.DataFrame, save_path) -> pd.DataFrame:
    """
    Apply all preprocessing steps to the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.
    
    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    df = preprocessing_volume_and_buy_qty(df)
    df = preprocessing_normalize(df)
    df = preprocessing_timestamp(df)
    save_to_parquet(df, save_path)
    return df
# for testing
# df = load_parquet("../dataset/train.parquet")
# df = preprocessing_volume_and_buy_qty(df)
# df = preprocessing_normalize(df)
# df = preprocessing_timestamp(df)
# print(df.head())
# # print(df['timestamp'].dtype)
# print(df.describe())
# save_to_parquet(df, "../dataset/train_processed.parquet")