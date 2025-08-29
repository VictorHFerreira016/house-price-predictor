# main.py
import argparse
from src.train import train_model
from src.evaluate import evaluate_model
from src.config import MODEL_CONFIG, PATH_CONFIG

def main(mode):
    model_file = PATH_CONFIG.MODEL_PATH / MODEL_CONFIG.MODEL_NAME
    features_file = PATH_CONFIG.MODEL_PATH / "model_features.txt"

    if mode == 'train':
        print("\n--- INICIANDO TREINAMENTO DO MODELO --- \n")
        train_model()

    elif mode == 'evaluate':
        print("\n--- AVALIANDO MODELO COM DADOS DE TREINO --- \n")
        data_file_for_evaluation = PATH_CONFIG.RAW_DATA_PATH / "train.csv"
        evaluate_model(str(model_file), str(data_file_for_evaluation), str(features_file))

    elif mode == 'predict':
        print("\n--- GERANDO ARQUIVO DE SUBMISSÃO --- \n")
        data_file_for_submission = PATH_CONFIG.RAW_DATA_PATH / "test.csv"
        evaluate_model(str(model_file), str(data_file_for_submission), str(features_file))

    else:
        print("Modo inválido. Escolha 'train', 'evaluate' ou 'predict'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executar o pipeline de ML.")
    parser.add_argument('mode', choices=['train', 'evaluate', 'predict'], 
                        help="O modo de execução: treinar, avaliar ou gerar previsões.")

    args = parser.parse_args()
    main(args.mode)