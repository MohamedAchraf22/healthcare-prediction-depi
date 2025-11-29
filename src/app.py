from flask import Flask, request, jsonify

from model_api import predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    
    output = predict(data)
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
