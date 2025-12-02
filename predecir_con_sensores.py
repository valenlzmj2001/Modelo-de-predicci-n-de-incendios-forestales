# ======================================================
# 🔥 PREDICCIÓN CON DATOS DE SENSORES (EJEMPLO)
# ======================================================
import joblib
import pandas as pd
from datetime import datetime

# Cargar modelo entrenado
print("🔥 Cargando modelo entrenado...")
modelo = joblib.load('modelo_incendios.pkl')
metadata = joblib.load('modelo_metadata.pkl')

print(f"✅ Modelo cargado (entrenado: {metadata['fecha_entrenamiento'][:10]})")
print(f"   Features requeridos: {', '.join(metadata['sensor_features'])}")

# ======================================================
# EJEMPLO DE USO CON SENSORES SIMULADOS
# ======================================================
print("\n" + "="*60)
print("📡 SIMULACIÓN DE LECTURA DE SENSORES")
print("="*60)

# Datos que vendrían de tus sensores de Wokwi
# DHT11: temperatura y humedad
# MQ2: gases (no usado directamente, pero indica combustión)
# Anemómetro simulado: viento
# Precipitación: acumulada (histórico o API meteorológica)

datos_sensores = {
    # Valores de sensores físicos
    'temperatura_media': 32.5,      # °C - DHT11/DHT22
    'humedad_relativa': 35.0,       # % - DHT11/DHT22
    'viento_max': 18.0,             # km/h - Anemómetro (simulado)
    
    # Precipitación acumulada (guardar histórico o consultar API)
    'precipitacion_7d': 0.0,        # mm últimos 7 días
    'precipitacion_14d': 2.5,       # mm últimos 14 días
    'precipitacion_30d': 8.0,       # mm últimos 30 días
    
    # Variables temporales (calcular automáticamente)
    'mes': datetime.now().month,
    'dia_año': datetime.now().timetuple().tm_yday,
    'estacion_seca': 1 if datetime.now().month in [12,1,2,6,7,8] else 0
}

print("\n📊 Datos de entrada:")
for key, value in datos_sensores.items():
    print(f"   {key:.<30} {value}")

# ======================================================
# HACER PREDICCIÓN
# ======================================================
print("\n🤖 Realizando predicción...")

# Convertir a DataFrame (el modelo espera este formato)
df_sensor = pd.DataFrame([datos_sensores])

# Predecir
prediccion = modelo.predict(df_sensor)[0]
probabilidad = modelo.predict_proba(df_sensor)[0, 1]

# Mostrar resultado
print("\n" + "="*60)
if prediccion == 1:
    print("⚠️  ALERTA: RIESGO DE INCENDIO DETECTADO")
    print(f"   Probabilidad: {probabilidad*100:.1f}%")
    
    if probabilidad > 0.7:
        print("   🔴 Riesgo ALTO - Activar alerta inmediata")
    elif probabilidad > 0.5:
        print("   🟠 Riesgo MODERADO - Monitoreo constante")
    else:
        print("   🟡 Riesgo BAJO - Precaución")
else:
    print("✅ SIN RIESGO DE INCENDIO")
    print(f"   Probabilidad de incendio: {probabilidad*100:.1f}%")
    print("   Condiciones normales")

print("="*60)

# ======================================================
# FUNCIÓN REUTILIZABLE PARA INTEGRACIÓN CON WOKWI
# ======================================================
def predecir_incendio(temperatura, humedad, viento, precip_7d, precip_14d, precip_30d):
    """
    Función para integrar con sensores de Wokwi
    
    Parámetros:
        temperatura: float - Temperatura en °C (DHT11)
        humedad: float - Humedad relativa en % (DHT11)
        viento: float - Velocidad del viento en km/h
        precip_7d: float - Precipitación acumulada últimos 7 días (mm)
        precip_14d: float - Precipitación acumulada últimos 14 días (mm)
        precip_30d: float - Precipitación acumulada últimos 30 días (mm)
    
    Retorna:
        dict con 'riesgo' (bool), 'probabilidad' (float), 'nivel' (str)
    """
    # Calcular variables temporales automáticamente
    now = datetime.now()
    mes = now.month
    dia_año = now.timetuple().tm_yday
    estacion_seca = 1 if mes in [12,1,2,6,7,8] else 0
    
    # Preparar datos
    datos = {
        'temperatura_media': temperatura,
        'humedad_relativa': humedad,
        'viento_max': viento,
        'precipitacion_7d': precip_7d,
        'precipitacion_14d': precip_14d,
        'precipitacion_30d': precip_30d,
        'mes': mes,
        'dia_año': dia_año,
        'estacion_seca': estacion_seca
    }
    
    df = pd.DataFrame([datos])
    prediccion = modelo.predict(df)[0]
    probabilidad = modelo.predict_proba(df)[0, 1]
    
    # Determinar nivel de riesgo
    if probabilidad > 0.7:
        nivel = "ALTO"
    elif probabilidad > 0.5:
        nivel = "MODERADO"
    elif probabilidad > 0.3:
        nivel = "BAJO"
    else:
        nivel = "MUY BAJO"
    
    return {
        'riesgo': bool(prediccion),
        'probabilidad': round(probabilidad, 3),
        'nivel': nivel,
        'timestamp': now.isoformat()
    }

# Ejemplo de uso de la función
print("\n💡 Ejemplo de uso de la función:")
print("-" * 60)
resultado = predecir_incendio(
    temperatura=32.5,
    humedad=35.0,
    viento=18.0,
    precip_7d=0.0,
    precip_14d=2.5,
    precip_30d=8.0
)
print(f"Resultado: {resultado}")

