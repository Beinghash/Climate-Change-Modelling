from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load saved artifacts
model = joblib.load("Climate_Change_modelling.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")
tfidf = joblib.load("tfidf_df.pkl")   # not used directly but kept for consistency


@app.route("/")
def home():
    """Render frontend form"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input data as JSON
        data = request.get_json()

        # Convert to correct order of features
        input_features = []
        for col in columns:
            val = data.get(col, 0)  # default 0 if missing
            input_features.append(float(val))

        # Reshape input for model
        input_array = np.array(input_features).reshape(1, -1)

        # Scale input
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = int(model.predict(input_scaled)[0])   # ensure integer class

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)



# Make sure to install requirements (Paste the below line in terminal and hit enter) before you run this file "app2.py"
# pip install -r requirements.txt 

# Or manually pip install the following: 
# flask
# scikit-learn
# xgboost
# joblib
# pandas
# numpy
