# House Price Predictor 
<p> Hello! This is a Machine Learning project designed to predict real estate prices based on their characteristics. It uses a Random Forest Regressor model and is structured in a modular and easy-to-use way, with a complete pipeline that ranges from data preprocessing to forecast generation.

## Project Structure 
- 'src/': preprocessing and training scripts
- 'data/raw/': raw data files
- 'models/': training model files

## How to run
1. Clone the repository 
git clone https://github.com/seu-usuario/house-price-predictor.git
cd house-price-predictor

#### 2. Create a Virtual Enviroment <p>
python -m venv .venv <p>
source .venv/bin/activate        # Linux/Mac <p>
.venv\Scripts\activate           # Windows

#### 3. Install Requirements <p>
pip install -r requirements.txt

#### 4. Add the data file <p>
Place the train.csv file in data/raw/.

#### 5. Train the model <p>
python src/train.py

#### 6. Evaluate the model <p>
python src/evaluate.py

**Create a virtual environment**

- python -m venv .venv
- source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

Now, install all the necessary libraries with a single command:

- pip install -r requirements.txt

2. Preparing the data

Place your data files (train.csv and test.csv) in the data/raw/ folder. The structure should be exactly as shown in the "Project Structure" section.

**How to use**

- python main.py train
This command will load the data from data/raw/train.csv, preprocess it, train a new RandomForestRegressor model, and save the trained model to models/house_price_model.pkl.

- python main.py evaluate
This command uses the previously trained model to make predictions on the same training data (train.csv) and calculates performance metrics. It also generates evaluation graphs and saves them in the reports/figures/ folder.

- python main.py predict