from flask import Flask , render_template  , request
import mlflow
from preprocessing_utility import normalize_text
import dagshub 
import pickle

import dagshub
mlflow.set_tracking_uri('https://dagshub.com/saurav3k2/Mini-mlops-project.mlflow')
dagshub.init(repo_owner='saurav3k2', repo_name='Mini-mlops-project', mlflow=True)


vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

app = Flask(__name__)


# Load model from model registry and make prediction on the text

model_name = "my_model"
model_version = 3

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # Bow Convert sparse matrix to DataFrame
    features = vectorizer.transform([text])
    
    # prediction
    result = model.predict(features)

    # show
    return render_template('index.html', result=result[0])


    # show
    return  str(result[0])




app.run(debug=True)