import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

# --- Veriyi yükle ---
df = pd.read_csv("nuclear_data_generated.csv")

# --- Özellikler ve hedef değişken ---
features = ['temp', 'pressure', 'boron_ppm', 'glcm_contrast']
target = 'neutron_flux_critical'

# Normalize et
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# --- Zaman serisi veri hazırlığı ---
timesteps = 3
X, y = [], []

for i in range(len(df) - timesteps):
    X.append(df[features].iloc[i:i+timesteps].values)
    y.append(df[target].iloc[i+timesteps])

X = np.array(X)
y = np.array(y)

# --- LSTM Modeli ---
model = Sequential([
    Input(shape=(timesteps, len(features))),
    LSTM(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=30, verbose=1)

# --- Örnek tahmin ---
sample = X[-1].reshape(1, timesteps, len(features))
prediction = model.predict(sample)[0][0]

print(f"Tahmin edilen risk (0-1): {prediction:.2f}")
if prediction > 0.6:
    print("⚠️ Kritik seviye yaklaşıyor!")
else:
    print("✅ Sistem normal çalışıyor.")
