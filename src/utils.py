import pandas as pd

def predict_on_test_data(model, test_data_path: str):
    """
    Load test data and make predictions using the trained model.
    
    Parameters:
    model: Trained LightGBM model.
    test_data_path (str): Path to the test data file.
    """
    test_df = pd.read_parquet(test_data_path)
    
    # Preprocess the test data (same preprocessing as training data)
    X_test = test_df.drop(columns=['ID', 'label'])
    print(f"columns: {X_test.columns}, column count: {len(X_test.columns)}")
    # Make predictions
    predictions = model.predict(X_test)
    
    # only save predictions and id, then to csv
    test_df['prediction'] = predictions
    # resed index as id
    test_df.reset_index(drop=True, inplace=True)
    test_df = test_df[['ID', 'prediction']]  # Keep only ID and predictions
    
    # Save to csv
    test_df.to_csv('/home/jasonx62301/for_python/data_mining/project/dataset/test_predictions.csv', index=False)
    
    return test_df