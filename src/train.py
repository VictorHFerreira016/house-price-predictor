import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import load_and_preprocess_data, preprocess_for_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib

def train_model():
    data_path = r'C:\Users\Aluno\OneDrive\Desktop\PROJECTS\data\raw\train.csv'

    # Carrega os dados e aplica o pró-processamento inicial)
    df = load_and_preprocess_data(data_path)

    # Prepara o dataframe para o modelo
    X_full = preprocess_for_model(df.drop(columns=['SalePrice']), trained_columns=None)
    y_full = df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

    model_features = X_train.columns.tolist()

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    scores = cross_val_score(model, X_full, y_full, cv = 5, scoring="neg_root_mean_squared_error")
    print(f'Média do RMSE (CV): {-scores.mean()}')

    rmse = root_mean_squared_error(y_test, predictions)
    print(f'Root Mean Squared Error: {rmse ** 0.5}')

    model_path = r'C:\Users\Aluno\OneDrive\Desktop\PROJECTS\models\house_price_model.pkl'
    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')

    features_path = r'C:\Users\Aluno\OneDrive\Desktop\PROJECTS\models\model_features.txt'
    joblib.dump(model_features, features_path)
    print(f'Model features saved to {features_path}')

if __name__ == "__main__":
    train_model()