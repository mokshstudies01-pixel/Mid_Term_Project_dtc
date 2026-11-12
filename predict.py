import pickle
import pandas as pd
from flask import Flask, request, jsonify

with open('model_binary.bin', 'rb') as f: model_bin = pickle.load(f)
with open('model_multiclass.bin', 'rb') as f: model_multi = pickle.load(f)

app = Flask('predict')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df_input = pd.DataFrame([data])

    # Ensure feature alignment with training template
    expected_cols = ['air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 'torque_nm', 'tool_wear_min', 'type_L', 'type_M']
    df_encoded = pd.get_dummies(df_input).reindex(columns=expected_cols, fill_value=0)

    proba = float(model_bin.predict_proba(df_encoded)[0, 1])
    failure_class = str(model_multi.predict(df_encoded)[0])

    return jsonify({
        'failure_probability': proba,
        'predicted_failure_type': failure_class
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
