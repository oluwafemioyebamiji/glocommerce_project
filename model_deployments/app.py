from flask import Flask, jsonify, request
import joblib
import numpy as np

#initialise the flask app
app = Flask(__name__)

# import our model file
mymodel = joblib.load('logmodel.pkl')
scaler = joblib.load('scaler.pkl')
@app.route('/predict', methods=["POST"])
def predict():
    #get the request the user is sending
    req = request.get_json()
    #reshape our features
    
    finalfeature = np.array(req['features']).reshape(1,-1)
    scaledfeatures = scaler.transform(finalfeature)
    print(scaledfeatures)
    prediction = mymodel.predict(finalfeature)
    print(prediction.tolist())

    return jsonify(prediction.tolist())

if __name__ == "__main__":
    app.run(debug=True)

#Just adding this extra comment