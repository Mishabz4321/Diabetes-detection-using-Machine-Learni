
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler= joblib.load('std_scaler.bin')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_features_scaled= scaler.transform(final_features)
    prediction = model.predict(final_features_scaled)
    if prediction==0:
        output='Non-diabetic'
    else:
        output='Diabetic'


    return render_template('index.html', prediction_text='The Person is {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)