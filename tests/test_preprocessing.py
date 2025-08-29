# tests/test_preprocessing.py
import pandas as pd
import numpy as np
from src.preprocessing import handle_missing_values

def test_handle_missing_values_lotfrontage():
    # Criação de um DataFrame de teste
    data = {
        'Neighborhood': ['A', 'A', 'B', 'B'],
        'LotFrontage': [60, 70, np.nan, 80]
    }
    df_test = pd.DataFrame(data)

    # Mediana do 'Neighborhood' A é 65, do B é 80
    expected_output = [60.0, 70.0, 80.0, 80.0]

    # Aplicar a função
    df_processed = handle_missing_values(df_test)

    # Verificar se os valores ausentes foram preenchidos corretamente
    assert df_processed['LotFrontage'].tolist() == expected_output
    assert not df_processed['LotFrontage'].isnull().any()