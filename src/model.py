from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pandas as pd
from pytorch_tabnet import TabNetRegressor
import numpy as np
import test

# baseline model for cryptocurrency price prediction
def baseline_model(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Train a baseline model using LightGBM on the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    target_col (str): The name of the target column.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    model: Trained LightGBM model.
    X_test: Test features.
    y_test: Test target values.
    """

    # Split the data into features and target
    X = df.drop(columns=[target_col, 'timestamp', 'hour', 'minute', 'dayofweek', 'day', 'month', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'dayofweek_sin', 'dayofweek_cos'])  # Exclude target and timestamp
    y = df[target_col]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Create and train the LightGBM model
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate RMSE
    rmse = mean_squared_error(y_test, y_pred)
    
    print(f"RMSE on test set: {rmse}")
    
    return model, X_test, y_test

# Example usage:
# df = pd.read_parquet('/home/jasonx62301/for_python/data_mining/project/dataset/train_processed.parquet')
# model, X_test, y_test = baseline_model(df, target_col='label')

# load test data and predict


# Example usage for prediction
# test_data_path = '/home/jasonx62301/for_python/data_mining/project/dataset/test_processed.parquet'
# predicted_df = predict_on_test_data(model, test_data_path)
# predicted_df.to_parquet('/home/jasonx62301/for_python/data_mining/project/dataset/test_predictions.parquet', index=False)

def tabnet_model(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Train a TabNet model on the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    target_col (str): The name of the target column.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    model: Trained TabNet model.
    X_test: Test features.
    y_test: Test target values.
    """
    
    # Split the data into features and target
    X = df.drop(columns=target_col + ['label', 'timestamp', 'hour', 'minute', 'dayofweek', 'day', 'month'])  # Exclude target and timestamp
    y = df[target_col]
    # print(f"columns: {X.columns}, column count: {len(X.columns)}")
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Create and train the TabNet model
    model = TabNetRegressor(verbose=1)
    model.fit(X_train.values, y_train.values, max_epochs=100)
    
    # Predict on the test set
    y_pred = model.predict(X_test.values)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"RMSE on test set: {rmse}")
    
    return model, X_test, y_test

target_cols = [
    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "dayofweek_sin", "dayofweek_cos"
]

df = pd.read_parquet('/home/jasonx62301/for_python/data_mining/project/dataset/train_processed.parquet')
# Example usage for TabNet model
model, X_test, y_test = tabnet_model(df, target_col=target_cols)
# Example usage for prediction with TabNet
test_data_path = '/home/jasonx62301/for_python/data_mining/project/dataset/test_processed.parquet'
def predict_on_test_data_tabnet(model, test_data_path: str):
    """
    Load test data and make predictions using the trained TabNet model.
    
    Parameters:
    model: Trained TabNet model.
    test_data_path (str): Path to the test data file.
    """
    test_df = pd.read_parquet(test_data_path)
    
    # Preprocess the test data (same preprocessing as training data)
    X_test = test_df.drop(columns=['ID', 'label'])
    # print(f"columns: {X_test.columns}, column count: {len(X_test.columns)}")

    # Make predictions
    predictions = model.predict(X_test.values)
    
    # tabnet is used for predicting "hour_sin", "hour_cos", "minute_sin", "minute_cos", "dayofweek_sin", "dayofweek_cos"
    for i, col in enumerate(target_cols):
        test_df[col] = predictions[:, i]
    # Reset index as ID
    test_df.reset_index(drop=True, inplace=True)
    test_df = test_df[['ID'] + target_cols]  # Keep only ID and predictions
    # Save to parquet
    test_df.to_parquet('/home/jasonx62301/for_python/data_mining/project/dataset/test_predictions_tabnet.parquet', index=False)
    return test_df

test_data_path = '/home/jasonx62301/for_python/data_mining/project/dataset/test_processed.parquet'
# test test_parquet
predicted_df_tabnet = predict_on_test_data_tabnet(model, test_data_path)
# Save the predictions to parquet
    