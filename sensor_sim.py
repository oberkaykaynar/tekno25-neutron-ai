import csv
import random
import time
from datetime import datetime

# Fiziksel olarak anlamlı veri aralıklarını tanımlıyoruz
TEMP_RANGE = (270, 330)  # sıcaklık (°C)
BORON_RANGE = (0, 2500)  # bor yoğunluğu (ppm)
FLUX_RANGE = (1e12, 1e14)  # nötron akısı (n/cm²·s)

# Kaç veri üretileceğini belirliyoruz
NUM_SAMPLES = 100

# CSV dosyasını yazmak üzere açıyoruz
with open('sensor_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Başlık satırını yazıyoruz
    writer.writerow(["timestamp", "temperature_C", "boron_ppm", "neutron_flux"])

    for _ in range(NUM_SAMPLES):
        # Şu anki zamanı ISO formatında alıyoruz
        timestamp = datetime.now().isoformat()

        # Her bir parametre için rastgele ama fiziksel anlamlı bir değer üretiyoruz
        temperature = round(random.uniform(*TEMP_RANGE), 2)
        boron = round(random.uniform(*BORON_RANGE), 1)
        neutron_flux = round(random.uniform(*FLUX_RANGE), 2)

        # Veriyi CSV dosyasına yazıyoruz
        writer.writerow([timestamp, temperature, boron, neutron_flux])

        # Aynı veriyi konsola da yazıyoruz
        print(f"{timestamp} | Temp: {temperature}°C | B: {boron} ppm | Flux: {neutron_flux:.2e}")

        # Her 1 saniyede bir veri üretimiyle gerçek zamanlı sistem simülasyonu yapıyoruz
        time.sleep(1)
