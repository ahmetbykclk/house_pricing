from flask import Flask, render_template, request, jsonify
from xgboost import XGBRegressor
import pandas as pd
import joblib

# Create Flask app
app = Flask(__name__)

# Load the trained model
model_xgb = joblib.load("trained_model.pkl")

# Define the categorical columns and their categories
categorical_columns = {
    "bulundugu_kat": ["-2", "1", "2", "3", "4", "5", "7", "10", "Bahce Dubleks", "Bahce Kati", "Duz Giris", "Mustakil Kat", "Yuksek Giris", "Cati Dubleks"],
    "isitma_tipi": ["Kat kaloriferi", "Kombi dogalgaz", "Merkezi dogalgaz", "Yerden isitma"]
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        brut = float(request.json['brut'])
        yas = request.json['yas']
        if yas == '21 ve uzeri':
            yas = 21  # Set an appropriate value for the upper range
        else:
            if "-" in yas:
                yas = yas.split("-")
                yas = sum(map(int, yas)) // 2
            else:
                yas = int(yas)
        kat1 = int(request.json['kat1'])
        net = float(request.json['net'])
        oda = float(request.json['oda'].split("+")[0])
        kat2 = request.json['kat2']
        isitma = request.json['isitma']

        # Create a DataFrame with the input data
        yeni_veri = pd.DataFrame({
            "brut_metrekare": [brut],
            "binanin_yasi": [yas],
            "binanin_kat_sayisi": [kat1],
            "net_metrekare": [net],
            "oda_sayisi": [oda],
            "bulundugu_kat": [kat2],
            "isitma_tipi": [isitma]
        })

        # Apply one-hot encoding to the categorical columns
        yeni_veri_encoded = pd.get_dummies(yeni_veri, columns=categorical_columns.keys())

        # Reorder columns to match the training data
        yeni_veri_encoded = yeni_veri_encoded.reindex(columns=model_xgb.get_booster().feature_names, fill_value=0)

        pred = model_xgb.predict(yeni_veri_encoded)

        if pred < 0:
            pred = -1 * pred

        pred = int(pred[0])  # Extract a single element from the prediction array

        para_birimi = "TL"
        pred_with_para_birimi = str(pred) + " " + para_birimi

        return jsonify({'prediction': pred_with_para_birimi})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
