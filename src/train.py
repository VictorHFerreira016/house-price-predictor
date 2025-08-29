import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib

from src.preprocessing import load_and_preprocess_data, preprocess_for_model
from src.config import MODEL_CONFIG, PATH_CONFIG
from src.validation import validate_file_exists, DataValidationError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model():
    """
    Train the house price prediction model with centralized configuration.
    
    Raises:
        DataValidationError: If training fails due to data issues
    """
    try:
        # Define paths using centralized config
        data_path = PATH_CONFIG.RAW_DATA_PATH / "train.csv"
        model_path = PATH_CONFIG.MODEL_PATH / MODEL_CONFIG.MODEL_NAME
        features_path = PATH_CONFIG.MODEL_PATH / MODEL_CONFIG.FEATURES_NAME

        # Ensure directories exist, mkdir() creates them if they don't
        # parents=True allows creating nested directories
        # exist_ok=True allows not raising an error if the directory already exists 
        PATH_CONFIG.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        
        # Validate input file exists
        validate_file_exists(data_path, "Training dataset")
        
        logger.info("=== STARTING TRAINING ===")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        # The parameter is_training=True was created to apply specific preprocessing for training, if not it is a testing set
        df = load_and_preprocess_data(str(data_path), is_training=True)
        
        # Prepare features and target
        if 'SalePrice' not in df.columns:
            raise DataValidationError("Column 'SalePrice' not found in dataset")

        # Splitting targets and features.
        X_full = preprocess_for_model(df.drop(columns=['SalePrice']))
        y_full = df['SalePrice']
        
        logger.info(f"Dataset prepared: {X_full.shape[0]} samples, {X_full.shape[1]} features")
        
        # Split data using config
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, 
            test_size=MODEL_CONFIG.TEST_SIZE, 
            random_state=MODEL_CONFIG.RANDOM_STATE
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Get model features for later use
        model_features = X_train.columns.tolist()
        logger.info(f"Total model features: {len(model_features)}")
        
        # Initialize and train model using config
        logger.info("Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=MODEL_CONFIG.N_ESTIMATORS,
            random_state=MODEL_CONFIG.RANDOM_STATE,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(X_train, y_train)
        logger.info("Training completed!")

        # Evaluate model performance
        logger.info("Evaluating model performance...")

        # Cross-validation using config
        # cv=MODEL_CONFIG.CV_FOLDS gets the configuration for cross-validation folds, folds are used to evaluate the model's performance
        # the higher the number of folds, the more accurate the evaluation, but also the longer the training time
        cv_scores = cross_val_score(
            model, X_full, y_full, 
            cv=MODEL_CONFIG.CV_FOLDS, 
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        # cv_rmse is the root mean squared error from cross-validation, it is negative because we used "neg_root_mean_squared_error" as the scoring method
        cv_rmse = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        logger.info(f"Cross-validation RMSE: {cv_rmse:.2f} (+/- {cv_std * 2:.2f})")
        
        # Test set evaluation
        predictions = model.predict(X_test)
        test_rmse = root_mean_squared_error(y_test, predictions)
        
        logger.info(f"Test set RMSE: {test_rmse:.2f}")
        
        # Check for overfitting, abs() is used to get the absolute difference, absolute means no sign
        if abs(cv_rmse - test_rmse) > cv_rmse * 0.1:  # 10% threshold
            logger.warning("Possible overfitting detected - significant difference between CV and test RMSE")
        else:
            logger.info("Model appears to be well generalized")

        # Save model
        logger.info("Saving model and features...")
        joblib.dump(model, model_path)
        joblib.dump(model_features, features_path)

        logger.info(f"Model saved at: {model_path}")
        logger.info(f"Features saved at: {features_path}")
        
        # Performance summary
        logger.info("=== TRAINING SUMMARY ===")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Features: {len(model_features)}")
        logger.info(f"CV RMSE: {cv_rmse:.2f}")
        logger.info(f"Test RMSE: {test_rmse:.2f}")
        logger.info(f"Model: Random Forest ({MODEL_CONFIG.N_ESTIMATORS} trees)")

        return {
            'cv_rmse': cv_rmse,
            'test_rmse': test_rmse,
            'n_features': len(model_features),
            'model_path': str(model_path)
        }
        
    except DataValidationError as e:
        logger.error(f"Data validation error during training: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        raise DataValidationError(f"Training failed: {str(e)}")

def main():
    """Main function to run training with error handling."""
    try:
        results = train_model()
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        return results
    except DataValidationError as e:
        logger.error(f"TRAINING FAILED: {str(e)}")
        return None
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return None

if __name__ == "__main__":
    main()