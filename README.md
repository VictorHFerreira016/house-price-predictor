# House Price Predictor 
<p>A Machine Learning project that predicts home values based on a variety of characteristics, such as square footage, neighborhood, basement quality, and more.

## Project Structure 
- 'src/': preprocessing and training scripts
- 'data/': data files
- 'models/': training model files

## How to run
#### 1. Clone the repository <p>
git clone https://github.com/seu-usuario/house-price-predictor.git <p>
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


