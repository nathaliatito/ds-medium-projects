import pandas as pd
import pickle
from flask import Flask, request
from data_preparing import run_scaler

model = pickle.load(open("model.pickle","rb"))
legend_dict = pickle.load(open("legend_dict.pickle", "rb"))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict():

    data_json = request.get_json(force=True)
    df_raw = pd.DataFrame(data_json)
    
    pipeline = run_scaler()
    df = pipeline.data_preparation(df_raw)

    X = df.values
    predict = model.predict(X)

    for k in legend_dict.keys():
        if predict[0] == k:
            resposta = legend_dict[k]

    return resposta

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
