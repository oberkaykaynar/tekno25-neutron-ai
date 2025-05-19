import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- Yeni veri setini yükle ---
df = pd.read_csv("nuclear_data_generated.csv")

# --- Giriş ve hedef değişkenleri belirle ---
features = ['temp', 'pressure', 'boron_ppm', 'glcm_contrast']
target = 'neutron_flux_critical'  # Kritik nötron akısı durumu (0 veya 1)

# X ve y hazırla
X = df[features]
y = df[target]

# --- Eğitim ve test setine ayır (80% eğitim - 20% test) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Random Forest modelini oluştur ve eğit ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Başarıyı değerlendir ---
y_pred = model.predict(X_test)
print("\n--- Model Doğruluğu ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

print("\n--- Sınıflandırma Raporu ---")
print(classification_report(y_test, y_pred))

# --- Anlık tahmin (son veriye göre) ---
# Özellik isimlerini koruyarak tek satırlık DataFrame oluştur
sample = pd.DataFrame([X.iloc[-1]], columns=X.columns)
prediction = model.predict(sample)[0]
print(f"\nAnlık tahmin: {'⚠️ Kritik' if prediction == 1 else '✅ Normal'}")
