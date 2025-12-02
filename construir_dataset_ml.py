# ======================================================
# 🔥 CONSTRUCCIÓN DE DATASET PARA ML - PREDICCIÓN DE INCENDIOS
# ======================================================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time

# --- CONFIGURACIÓN ---
FIRMS_CALI_CSV = "FIRMS_Cali.csv"
OUTPUT_CSV = "dataset_incendios_ml.csv"

# Bounding box de Cali (cerros)
LAT_MIN, LAT_MAX = 3.30, 3.55
LON_MIN, LON_MAX = -76.65, -76.45

# Rango temporal
START_DATE = "2012-01-01"
END_DATE = "2025-12-31"

# Grid espacial para muestras negativas (celdas de ~500m)
GRID_SIZE = 0.005  # ~500m en grados

# Ratio positivos:negativos
NEGATIVE_RATIO = 5  # 5 muestras negativas por cada positiva

print("🔥 CONSTRUCCIÓN DE DATASET PARA ML - PREDICCIÓN DE INCENDIOS")
print("=" * 60)

# ======================================================
# 1. CARGAR EVENTOS POSITIVOS (INCENDIOS CONFIRMADOS)
# ======================================================
print("\n📂 Cargando eventos de incendio desde FIRMS...")
fires = pd.read_csv(FIRMS_CALI_CSV)
fires['acq_date'] = pd.to_datetime(fires['acq_date'], errors='coerce')
fires = fires.dropna(subset=['acq_date', 'latitude', 'longitude'])
fires = fires.sort_values('acq_date')

print(f"✅ {len(fires)} eventos de incendio cargados (2012-2025)")
print(f"   Rango: {fires['acq_date'].min()} → {fires['acq_date'].max()}")

# ======================================================
# 2. CREAR GRID ESPACIAL Y DÍAS ÚNICOS
# ======================================================
print("\n🗺️  Creando grid espacial y eventos diarios...")

# Discretizar ubicaciones en grid
fires['grid_lat'] = (fires['latitude'] / GRID_SIZE).round() * GRID_SIZE
fires['grid_lon'] = (fires['longitude'] / GRID_SIZE).round() * GRID_SIZE
fires['date_only'] = fires['acq_date'].dt.date

# Agrupar por día y celda (un evento = 1 incendio/día/celda)
eventos_positivos = fires.groupby(['date_only', 'grid_lat', 'grid_lon']).agg({
    'brightness': 'mean',
    'frp': 'mean',
    'confidence': 'first'
}).reset_index()
eventos_positivos['incendio'] = 1

print(f"✅ {len(eventos_positivos)} eventos únicos (día × celda)")

# ======================================================
# 3. GENERAR MUESTRAS NEGATIVAS (DÍAS SIN INCENDIO)
# ======================================================
print("\n🔄 Generando muestras negativas...")

# Crear grid completo
lats = np.arange(LAT_MIN, LAT_MAX, GRID_SIZE)
lons = np.arange(LON_MIN, LON_MAX, GRID_SIZE)
grid_cells = [(round(lat, 4), round(lon, 4)) for lat in lats for lon in lons]

# Rango de fechas
date_range = pd.date_range(START_DATE, END_DATE, freq='D')

# Muestrear aleatoriamente
np.random.seed(42)
n_negatives = len(eventos_positivos) * NEGATIVE_RATIO
negative_samples = []

for _ in range(n_negatives):
    date = pd.Timestamp(np.random.choice(date_range)).date()
    lat, lon = grid_cells[np.random.randint(len(grid_cells))]
    
    # Verificar que no sea un evento positivo
    if not ((eventos_positivos['date_only'] == date) & 
            (eventos_positivos['grid_lat'] == lat) & 
            (eventos_positivos['grid_lon'] == lon)).any():
        negative_samples.append({
            'date_only': date,
            'grid_lat': lat,
            'grid_lon': lon,
            'incendio': 0
        })

eventos_negativos = pd.DataFrame(negative_samples)
print(f"✅ {len(eventos_negativos)} muestras negativas generadas")

# ======================================================
# 4. COMBINAR POSITIVOS Y NEGATIVOS
# ======================================================
print("\n🔗 Combinando eventos...")
dataset = pd.concat([
    eventos_positivos[['date_only', 'grid_lat', 'grid_lon', 'incendio']],
    eventos_negativos
], ignore_index=True)

dataset = dataset.sort_values('date_only').reset_index(drop=True)
print(f"✅ Dataset base: {len(dataset)} registros")
print(f"   Positivos: {(dataset['incendio']==1).sum()} ({(dataset['incendio']==1).sum()/len(dataset)*100:.1f}%)")
print(f"   Negativos: {(dataset['incendio']==0).sum()} ({(dataset['incendio']==0).sum()/len(dataset)*100:.1f}%)")

# ======================================================
# 5. DESCARGAR DATOS METEOROLÓGICOS (OPEN-METEO ARCHIVE)
# ======================================================
print("\n🌦️  Descargando datos meteorológicos históricos...")
print("   (Esto puede tardar varios minutos...)")

def get_weather_data(lat, lon, start_date, end_date):
    """Descarga datos meteorológicos de Open-Meteo Archive API"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                 "relative_humidity_2m_mean,precipitation_sum,wind_speed_10m_max",
        "timezone": "America/Bogota"
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if 'daily' in data:
                df = pd.DataFrame(data['daily'])
                df['date'] = pd.to_datetime(df['time']).dt.date
                return df
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    return None

# Obtener datos meteorológicos por ubicación única
unique_locations = dataset[['grid_lat', 'grid_lon']].drop_duplicates()
weather_cache = {}

for idx, (lat, lon) in enumerate(unique_locations.values):
    if idx % 10 == 0:
        print(f"   Progreso: {idx}/{len(unique_locations)} ubicaciones...")
    
    # Rango de fechas para esta ubicación
    loc_events = dataset[(dataset['grid_lat']==lat) & (dataset['grid_lon']==lon)]
    min_date = loc_events['date_only'].min()
    max_date = loc_events['date_only'].max()
    
    weather_df = get_weather_data(lat, lon, str(min_date), str(max_date))
    if weather_df is not None:
        weather_cache[(lat, lon)] = weather_df
    
    time.sleep(0.5)  # Rate limiting

print(f"✅ Datos meteorológicos descargados para {len(weather_cache)} ubicaciones")

# ======================================================
# 6. UNIR DATOS METEOROLÓGICOS CON EVENTOS
# ======================================================
print("\n🔗 Uniendo datos meteorológicos con eventos...")

# Función para calcular precipitación acumulada
def get_precip_accumulated(weather_df, target_date, days):
    """Calcula precipitación acumulada N días antes de target_date"""
    target = pd.to_datetime(target_date)
    start = target - timedelta(days=days)
    mask = (pd.to_datetime(weather_df['date']) >= start) & (pd.to_datetime(weather_df['date']) < target)
    return weather_df.loc[mask, 'precipitation_sum'].sum()

# Añadir features meteorológicos
weather_features = []
for idx, row in dataset.iterrows():
    if idx % 500 == 0:
        print(f"   Procesando: {idx}/{len(dataset)} registros...")
    
    lat, lon = row['grid_lat'], row['grid_lon']
    date = row['date_only']
    
    if (lat, lon) in weather_cache:
        weather_df = weather_cache[(lat, lon)]
        day_weather = weather_df[weather_df['date'] == date]
        
        if not day_weather.empty:
            features = {
                'temperatura_max': day_weather['temperature_2m_max'].values[0],
                'temperatura_min': day_weather['temperature_2m_min'].values[0],
                'temperatura_media': day_weather['temperature_2m_mean'].values[0],
                'humedad_relativa': day_weather['relative_humidity_2m_mean'].values[0],
                'viento_max': day_weather['wind_speed_10m_max'].values[0],
                'precipitacion_dia': day_weather['precipitation_sum'].values[0],
                'precipitacion_7d': get_precip_accumulated(weather_df, date, 7),
                'precipitacion_14d': get_precip_accumulated(weather_df, date, 14),
                'precipitacion_30d': get_precip_accumulated(weather_df, date, 30),
            }
        else:
            features = {k: np.nan for k in [
                'temperatura_max', 'temperatura_min', 'temperatura_media',
                'humedad_relativa', 'viento_max', 'precipitacion_dia',
                'precipitacion_7d', 'precipitacion_14d', 'precipitacion_30d'
            ]}
    else:
        features = {k: np.nan for k in [
            'temperatura_max', 'temperatura_min', 'temperatura_media',
            'humedad_relativa', 'viento_max', 'precipitacion_dia',
            'precipitacion_7d', 'precipitacion_14d', 'precipitacion_30d'
        ]}
    
    weather_features.append(features)

weather_df_final = pd.DataFrame(weather_features)
dataset_final = pd.concat([dataset, weather_df_final], axis=1)

# Eliminar filas con NaN en variables críticas
dataset_final = dataset_final.dropna(subset=[
    'temperatura_media', 'humedad_relativa', 'precipitacion_7d'
])

print(f"✅ Dataset final: {len(dataset_final)} registros completos")

# ======================================================
# 7. AÑADIR FEATURES TEMPORALES
# ======================================================
print("\n📅 Añadiendo features temporales...")
dataset_final['fecha'] = pd.to_datetime(dataset_final['date_only'])
dataset_final['año'] = dataset_final['fecha'].dt.year
dataset_final['mes'] = dataset_final['fecha'].dt.month
dataset_final['dia_año'] = dataset_final['fecha'].dt.dayofyear
dataset_final['estacion_seca'] = dataset_final['mes'].isin([12, 1, 2, 6, 7, 8]).astype(int)

# ======================================================
# 8. GUARDAR DATASET FINAL
# ======================================================
print("\n💾 Guardando dataset final...")

# Reordenar columnas
cols_order = [
    'fecha', 'año', 'mes', 'dia_año', 'estacion_seca',
    'grid_lat', 'grid_lon',
    'temperatura_max', 'temperatura_min', 'temperatura_media',
    'humedad_relativa', 'viento_max',
    'precipitacion_dia', 'precipitacion_7d', 'precipitacion_14d', 'precipitacion_30d',
    'incendio'
]
dataset_final = dataset_final[cols_order]

dataset_final.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Dataset guardado: {OUTPUT_CSV}")
print(f"\n📊 RESUMEN FINAL:")
print(f"   Total registros: {len(dataset_final)}")
print(f"   Incendios (1): {(dataset_final['incendio']==1).sum()}")
print(f"   No incendios (0): {(dataset_final['incendio']==0).sum()}")
print(f"   Variables: {len(cols_order)}")
print(f"   Rango temporal: {dataset_final['fecha'].min()} → {dataset_final['fecha'].max()}")
print("\n🎯 Listo para entrenamiento de modelo ML!")
print("\n💡 Sugerencias:")
print("   - Reserva 2023-2025 para validación temporal")
print("   - Usa RandomForest o XGBoost como baseline")
print("   - Métricas: Precision, Recall, AUC-PR (por desequilibrio)")
print("   - Considera validación espacial cruzada")

