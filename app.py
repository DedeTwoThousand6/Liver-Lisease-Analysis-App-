import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# --- LOGIKA DARI IPYNB (Tanpa Merubah Logika) ---
def train_and_save_model():
    # Memuat dataset sesuai spesifikasi IPYNB
    dataset = pd.read_csv("Liver Patient Dataset (LPD)_train.csv", encoding='latin1')
    dataset.columns = dataset.columns.str.strip()
    dataset = dataset.fillna(dataset.median(numeric_only=True))
    
    gender_col = [col for col in dataset.columns if 'gender' in col.lower()][0]
    dataset[gender_col] = dataset[gender_col].fillna(dataset[gender_col].mode()[0])
    dataset[gender_col] = dataset[gender_col].map({'Male': 1, 'Female': 0})
    
    target_col = dataset.columns[-1]
    dataset[target_col] = dataset[target_col].map({1: 1, 2: 0})
    
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Arsitektur Backpropagation sesuai IPYNB
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_scaled, y, epochs=50, batch_size=32, verbose=0)
    
    # Simpan Model dan Scaler
    model.save('model_liver.h5')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return model, scaler

# Load model saat start-up
try:
    model = load_model('model_liver.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except:
    print("Melatih model pertama kali...")
    model, scaler = train_and_save_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil input dari form HTML
        input_features = [float(x) for x in request.form.values()]
        final_features = np.array([input_features])
        
        # Normalisasi menggunakan scaler yang sama saat training
        scaled_features = scaler.transform(final_features)
        
        # Prediksi
        prediction = model.predict(scaled_features)
        output = (prediction[0][0] > 0.5).astype(int)
        
        res_text = "Pasien Liver (Sakit)" if output == 1 else "Bukan Pasien Liver (Sehat)"
        probability = round(float(prediction[0][0]) * 100, 2) if output == 1 else round((1 - float(prediction[0][0])) * 100, 2)

        return render_template('index.html', 
                               prediction_text=res_text, 
                               prob=f"Tingkat Keyakinan: {probability}%",
                               status="danger" if output == 1 else "success")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", status="warning")

if __name__ == "__main__":
    app.run(debug=True)