import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error
import joblib
import os 
from src.preprocessing import load_and_preprocess_data, preprocess_for_model
from src.config import get_config, PATH_CONFIG

# This function is used to test and evaluate the model with data test.
def evaluate_model(model_path: str, data_path: str, features_path: str):
    # If the model file does not exist
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please train the model first.")
        return
    # If the features does not exist 
    if not os.path.exists(features_path):
        print(f"Features file not found at {features_path}. Please ensure the model features are saved.")
        return
    
    # joblib loads the model with model_path to the model variable
    model = joblib.load(model_path)
    # it loads the features columns to the trained_columns
    trained_columns = joblib.load(features_path)
    print(f"Model loaded from {model_path}")

    # It calls the load_and_preprocess_data function to do all preprocess data
    df_raw = load_and_preprocess_data(data_path)
    print(f"Data loaded from {data_path}")

    # It calls the function preprocess for model, and copies the df_raw to not modificate 
    # the original one
    X_processed = preprocess_for_model(df_raw.copy(), trained_columns=trained_columns)
    
    # It saves in the y_test (target) the target column 'SalePrice',
    # and saves in X_test the processed features
    if 'SalePrice' in df_raw.columns:
        y_test = df_raw['SalePrice']
        X_test = X_processed

        # Makes predictions using X_test, as that, the preprocessed features
        predictions = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        msle = mean_squared_log_error(y_test, predictions)

        print(f'Root Mean Squared Error: {rmse}')
        print(f'R^2 Score: {r2}')
        print(f'Mean Absolute Error: {mae}')
        print(f'Mean Squared Log Error: {msle}')

        try: 
            import matplotlib.pyplot as plt
            import seaborn as sns
            # Defines the folder path to save the figures
            output_dir = PATH_CONFIG.FIGURES_PATH
            # Creates a folder caled "reports/figures"
            os.makedirs(output_dir, exist_ok=True)

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=predictions)
            # Adds a diagonal line representing perfect predictions, indicating where the predicted values would equal the actual values
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
            plt.xlabel("Actual Prices")
            plt.ylabel("Predicted Prices")
            plt.title("Actual vs Predicted Prices")
            # Saves the figures using savefig function
            plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"))
            plt.close()

            errors = y_test - predictions
            plt.figure(figsize=(10, 6))
            sns.histplot(errors, kde=True)
            plt.xlabel("Prediction Errors")
            plt.title("Distribution of Prediction Errors")
            plt.savefig(os.path.join(output_dir, "prediction_errors.png"))
            plt.close()

            print(f"\nPlots saved to {output_dir}")

        except ImportError:
            print("Matplotlib or Seaborn not installed. Skipping plot generation.")

    else:
        # These lines are responsable to generate a submission file
        X_test = X_processed
        predictions = model.predict(X_test)

        # Creates a DataFrame with 2 columns.
        submission_df = pd.DataFrame({
            'Id': df_raw['Id'] if 'Id' in df_raw.columns else df_raw.index,
            'SalePrice': predictions
        })
        # Saves the DataFrame to a CSV file without the index
        # The submission file is used for making predictions on the test set

        output_path = PATH_CONFIG.PROCESSED_DATA_PATH / "submission.csv"
        submission_df.to_csv(output_path, index=False)
        print(f"Submission file saved to {output_path}")

if __name__ == "__main__":
    configs = get_config()
    model_file = configs['paths'].MODEL_PATH / "house_price_model.pkl"
    features_file = configs['paths'].MODEL_PATH / "model_features.txt"
    data_file_for_evaluation = configs['paths'].RAW_DATA_PATH / "train.csv"
    data_file_for_submission = configs['paths'].RAW_DATA_PATH / "test.csv"

    print("\n --- Evaluating Model --- \n")
    evaluate_model(str(model_file), str(data_file_for_evaluation), str(features_file))
    
    print("\n --- Generating Submission File --- \n")
    evaluate_model(str(model_file), str(data_file_for_submission), str(features_file))