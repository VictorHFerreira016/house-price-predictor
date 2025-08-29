from dataclasses import dataclass
from pathlib import Path

# @dataclass is a decorator that automatically generates special methods for classes,
# such as __init__(), __repr__(), and __eq__(), based on the class attributes.
# So the classes ModelConfig, DataConfig, PathConfig, and PlotConfig don't need a constructor.
@dataclass
class ModelConfig:
    """Configs related to the ML model"""
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    N_ESTIMATORS: int = 100
    MODEL_NAME: str = "house_price_model.pkl"
    FEATURES_NAME: str = "model_features.txt"

@dataclass
class DataConfig:
    """Configs related to the data"""
    # Columns that should be filled with "None" when missing
    NONE_FILL_COLUMNS: list[str] | None = None

    # Numeric columns that should be filled with 0
    ZERO_FILL_COLUMNS: list[str] | None = None

    # Required columns in the training dataset
    REQUIRED_TRAIN_COLUMNS: list[str] | None = None

    # Required columns in the test dataset
    REQUIRED_TEST_COLUMNS: list[str] | None = None

    # Columns for feature engineering
    TOTAL_AREA_COLUMNS: list[str] | None = None

    def __post_init__(self):
        """Initializes the lists after the instance is created"""
        if self.NONE_FILL_COLUMNS is None:
            self.NONE_FILL_COLUMNS = [
                "PoolQC", "Alley", "Fence", "FireplaceQu", "GarageQual", 
                "GarageType", "GarageFinish", "GarageCond", "BsmtExposure", 
                "BsmtFinType1", "BsmtFinType2", "BsmtQual", "BsmtCond", 
                "MasVnrType", "MiscFeature"
            ]
        
        if self.ZERO_FILL_COLUMNS is None:
            self.ZERO_FILL_COLUMNS = ["MasVnrArea", "GarageYrBlt"]
        
        if self.REQUIRED_TRAIN_COLUMNS is None:
            self.REQUIRED_TRAIN_COLUMNS = [
                "SalePrice", "LotArea", "OverallQual", "OverallCond", 
                "YearBuilt", "GrLivArea", "Neighborhood"
            ]
        
        if self.REQUIRED_TEST_COLUMNS is None:
            self.REQUIRED_TEST_COLUMNS = [
                "Id", "LotArea", "OverallQual", "OverallCond", 
                "YearBuilt", "GrLivArea", "Neighborhood"
            ]
        
        if self.TOTAL_AREA_COLUMNS is None:
            self.TOTAL_AREA_COLUMNS = [
                "1stFlrSF", "2ndFlrSF", "LowQualFinSF", 
                "GrLivArea", "BsmtFinSF1", "BsmtFinSF2"
            ]

class PathConfig:
    """Configs related to paths"""
    def __init__(self):
        self.PROJECT_ROOT = self._get_project_root()
        self.RAW_DATA_PATH = self.PROJECT_ROOT / "data" / "raw"
        self.PROCESSED_DATA_PATH = self.PROJECT_ROOT / "data" / "processed"
        self.MODEL_PATH = self.PROJECT_ROOT / "models"
        self.REPORTS_PATH = self.PROJECT_ROOT / "reports"
        self.FIGURES_PATH = self.REPORTS_PATH / "figures"
        
    def _get_project_root(self) -> Path:
        """Returns the root directory of the project"""
        return Path(__file__).parent.parent

@dataclass
class PlotConfig:
    """Configs related to plot generation"""
    FIGURE_SIZE: tuple = (10, 6)
    DPI: int = 300
    STYLE: str = "seaborn-v0_8"
    SAVE_FORMAT: str = "png"

# Config global instances
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
PATH_CONFIG = PathConfig()
PLOT_CONFIG = PlotConfig()

# Convenience function for accessing settings
def get_config():
    """Returns all configurations as a dictionary"""
    return {
        'model': MODEL_CONFIG,
        'data': DATA_CONFIG,
        'paths': PATH_CONFIG,
        'plots': PLOT_CONFIG
    }