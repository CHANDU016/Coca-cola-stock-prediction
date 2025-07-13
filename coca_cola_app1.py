from flask import Flask, render_template, request, render_template_string, redirect, url_for
import os
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import numpy as np

Data1 = None

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = 'Data.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Redirect to /data1 after successful upload
            return redirect(url_for('data_cleaning'))

    return render_template('index.html')


@app.route('/data1')
def data_cleaning():
    global Data1
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'Data.csv')

    if not os.path.exists(filepath):
        return "No uploaded data found."

    Data1 = pd.read_csv(filepath)

    # Perform cleaning
    Data1['MA_20'] = Data1['Close'].rolling(window=20).mean()
    Data1['MA_50'] = Data1['Close'].rolling(window=50).mean()
    Data1['Daily_Return'] = Data1['Close'].pct_change()
    Data1['Volatility'] = Data1['Daily_Return'].rolling(window=20).std()
    Data1.dropna(inplace=True)

    table_html = Data1.to_html(classes='table table-striped', index=False)

    return render_template('table.html')


import pickle
@app.route('/predict')
def predict():
    global Data1
    try:

        if Data1 is None:
            return "❌ Data not loaded. Please upload and clean the data first at '/' and '/data1'."
        # Load model
        with open("stock_rmodel.pkl", "rb") as f:
            model = pickle.load(f)

        # Preprocess data
        close_data = Data1['Close'].values.reshape(-1, 1)
        scaler_x = MinMaxScaler()
        scaled_data = scaler_x.fit_transform(close_data)

        # Prepare sequences
        X = []
        seq_length = 10
        for i in range(seq_length, len(scaled_data)):
            X.append(scaled_data[i - seq_length:i, 0])
        X = np.array(X)

        # Reshape for prediction (2D input for sklearn models)
        X = X.reshape(X.shape[0], X.shape[1])

        # Predict
        y_pred_scaled = model.predict(X)

        # Inverse transform prediction
        scaler_y = MinMaxScaler()
        scaler_y.fit(close_data[seq_length:])
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

        # Convert to DataFrame
        result_df = pd.DataFrame({
            'Actual': close_data[seq_length:].flatten(),
            'Predicted': y_pred.flatten()
        })

        # Show in web page
        table_html = result_df.to_html(classes='table table-striped', index=False)

        return render_template_string('''
            <html>
                <head>
                    <title>Prediction Results</title>
                    <style>
                        body { font-family: Arial; padding: 20px; }
                        table { border-collapse: collapse; width: 100%; }
                        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                    </style>
                </head>
                <body>
                    <h2>📈 Prediction Results</h2>
                    {{ table|safe }}
                </body>
            </html>
        ''', table=table_html)

    except Exception as e:
        return f"❌ Error in prediction: {str(e)}"



if __name__ == '__main__':
    app.run(debug=True)
