import pandas as pd
from typing import Optional, Dict, Any
import logging
from pathlib import Path

# logger is used for logging validation errors and information
# getLogger is called to create a logger instance
# __name__ is used to get the name of the current module
# Example: a module named "my_module" would have a logger named "my_module"
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Customized exception for data validation errors"""
    pass

class SchemaValidator:
    """Class for validating DataFrame schemas"""

    # staticmethod decorator is used to define a static method, it means that the method 
    # can be called on the class itself, rather than on an instance of the class.
    @staticmethod
    def validate_dataframe_not_empty(df: pd.DataFrame, name: str = "DataFrame") -> None:
        """
        Validates that the DataFrame is not empty.

        Args:
            df: DataFrame to validate
            name: Name of the DataFrame for error messages

        Raises:
            DataValidationError: If the DataFrame is emp tty
        """
        if df.empty:
            raise DataValidationError(f"{name} is empty")

        logger.info(f"{name} contains {len(df)} rows and {len(df.columns)} columns")

    @staticmethod
    def validate_required_columns(df: pd.DataFrame, required_columns: list[str], 
                                name: str = "DataFrame") -> None:
        """
        Validate if all required columns are present in the DataFrame.

        Args:
            df: DataFrame to validate
            required_columns: List of required columns
            name: Name of the DataFrame for error messages

        Raises:
            DataValidationError: If any required column is missing
        """
        if not required_columns:
            logger.warning(f"No required columns specified for {name}")
            return

        # Find missing columns, it gets the required_columns columns and gets the same that is in df.columns, if it is missing, it's gonna be
        # considered as missing
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            raise DataValidationError(
                f"{name} is missing required columns: {sorted(missing_columns)}"
            )

        logger.info(f"{name} contains all {len(required_columns)} required columns")

    @staticmethod
    def validate_column_types(df: pd.DataFrame, column_types: Dict[str, str], 
                            name: str = "DataFrame") -> None:
        """
        Validates column types in the DataFrame.

        Args:
            df: DataFrame to validate
            column_types: Dictionary {column: expected_type}
            name: Name of the DataFrame for error messages

        Raises:
            DataValidationError: If any column does not have the expected type
        """

        # Initialize a list to collect type errors
        type_errors = []

        # column_types is a dictionary mapping column names to their expected types, items() gets the key-value pairs.
        for column, expected_type in column_types.items():
            if column not in df.columns:
                continue  # Column does not exist, will be validated elsewhere

            actual_type = str(df[column].dtype)

            # Flexible type mapping
            type_mapping = {
                'numeric': ['int64', 'int32', 'float64', 'float32'],
                'string': ['object', 'string'],
                'categorical': ['object', 'category'],
                'datetime': ['datetime64[ns]', 'datetime64']
            }

            # Check if the expected type is in the mapping
            if expected_type in type_mapping:
                if actual_type not in type_mapping[expected_type]:
                    type_errors.append(f"'{column}': waiting {expected_type}, found {actual_type}")
            else:
                if actual_type != expected_type:
                    type_errors.append(f"'{column}': waiting {expected_type}, found {actual_type}")
        
        if type_errors:
            raise DataValidationError(f"{name} has incorrect types: {'; '.join(type_errors)}")

        logger.info(f"{name} passed type validation.")

    @staticmethod
    def validate_no_duplicates(df: pd.DataFrame, columns: Optional[list[str]] = None, 
                             name: str = "DataFrame") -> None:
        """
        Valida se não há duplicatas no DataFrame.
        
        Args:
            df: DataFrame para validar
            columns: Lista de colunas para verificar duplicatas (None = todas)
            name: Nome do DataFrame para mensagens de erro
            
        Raises:
            DataValidationError: Se houver duplicatas
        """
        if columns:
            duplicates = df.duplicated(subset=columns).sum()
            context = f"nas colunas {columns}"
        else:
            duplicates = df.duplicated().sum()
            context = "em todas as colunas"
        
        if duplicates > 0:
            raise DataValidationError(f"{name} contém {duplicates} duplicatas {context}")
        
        logger.info(f"{name} não contém duplicatas {context}")

    @staticmethod
    def validate_value_ranges(df: pd.DataFrame, value_ranges: Dict[str, Dict[str, Any]], 
                            name: str = "DataFrame") -> None:
        """
        Valida se os valores das colunas estão dentro dos ranges esperados.
        
        Args:
            df: DataFrame para validar
            value_ranges: Dict com {coluna: {'min': valor, 'max': valor}}
            name: Nome do DataFrame para mensagens de erro
            
        Raises:
            DataValidationError: Se algum valor estiver fora do range
        """
        range_errors = []
        
        for column, ranges in value_ranges.items():
            if column not in df.columns:
                continue
            
            series = df[column]
            
            if 'min' in ranges:
                min_val = ranges['min']
                if series.min() < min_val:
                    range_errors.append(f"'{column}': valor mínimo {series.min()} < {min_val}")
            
            if 'max' in ranges:
                max_val = ranges['max']
                if series.max() > max_val:
                    range_errors.append(f"'{column}': valor máximo {series.max()} > {max_val}")
        
        if range_errors:
            raise DataValidationError(f"{name} tem valores fora do range: {'; '.join(range_errors)}")
        
        logger.info(f"{name} passou na validação de ranges")

    @staticmethod
    def validate_missing_values_threshold(df: pd.DataFrame, threshold: float = 0.5, 
                                        name: str = "DataFrame") -> list[str]:
        """
        Valida quais colunas têm muitos valores ausentes.
        
        Args:
            df: DataFrame para validar
            threshold: Limite máximo de valores ausentes (0.0 a 1.0)
            name: Nome do DataFrame para mensagens de erro
            
        Returns:
            Lista de colunas com muitos valores ausentes
        """
        missing_pct = df.isnull().sum() / len(df)
        problematic_cols = missing_pct[missing_pct > threshold].index.tolist()
        
        if problematic_cols:
            logger.warning(f"{name} tem {len(problematic_cols)} colunas com >{threshold*100}% valores ausentes: {problematic_cols}")
        else:
            logger.info(f"{name} não tem colunas com >{threshold*100}% valores ausentes")
        
        return problematic_cols

def validate_file_exists(file_path: str | Path, description: str = "Arquivo") -> None:
    """
    Validate if the file exists.

    Args:
        file_path: File path.
        description: File description for error messages.

    Raises:
        DataValidationError: If the file does not exist
    """

    path = Path(file_path)
    if not path.exists():
        raise DataValidationError(f"{description} not found: {file_path}")
    
    if not path.is_file():
        raise DataValidationError(f"{description} is not a valid file: {file_path}")

    logger.info(f"{description} found: {file_path}")

def validate_train_dataset(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """
    Convenience function to validate training dataset.

    Args:
        df: Training DataFrame
        required_columns: Required columns

    Returns:
        Validated DataFrame

    Raises:
        DataValidationError: If any validation fails
    """

    # Let's create an instance of the class
    validator = SchemaValidator()
    
    # Basics validations
    validator.validate_dataframe_not_empty(df, "Training dataset")
    validator.validate_required_columns(df, required_columns, "Training dataset")

    # Specific validations for training
    if 'SalePrice' in df.columns:
        # SalePrice must be positive
        if (df['SalePrice'] <= 0).any():
            raise DataValidationError("Training dataset contains non-positive values in SalePrice")

    # Check for excessive missing values, in this case if 80% or more are missing
    problematic_cols = validator.validate_missing_values_threshold(df, 0.8, "Training dataset")

    logger.info("Training dataset passed all validations")
    return df

def validate_test_dataset(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """
    Convenience function to validate test dataset.

    Args:
        df: Test DataFrame
        required_columns: Required columns

    Returns:
        Validated DataFrame

    Raises:
        DataValidationError: If any validation fails
    """
    
    # It does the same as validate_train_dataset
    validator = SchemaValidator()

    # Basics validations
    validator.validate_dataframe_not_empty(df, "Test dataset")
    validator.validate_required_columns(df, required_columns, "Test dataset")

    # ID must be unique
    if 'Id' in df.columns:
        validator.validate_no_duplicates(df, ['Id'], "Test dataset")

    logger.info("Test dataset passed all validations")
    return df