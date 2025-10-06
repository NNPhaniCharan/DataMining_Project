from flask import Flask, jsonify
import joblib

app = Flask(__name__)

@app.route('/clustering', methods=['GET'])
def get_clusters():
    # Load trained clustering model
    model = joblib.load("Backend/models/kmeans_custom_model.joblib")

    # Prepare response
    response = {
        "best_k": model.k,
        "centroids": model.centroids.tolist()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
