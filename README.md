# 🔥 Sistema de Predicción de Incendios - Cali, Colombia

Sistema de Machine Learning para predecir riesgo de incendios forestales en los cerros de Cali basado en datos históricos de FIRMS (NASA) y condiciones meteorológicas.

---

## 📁 Estructura del Proyecto

```
Incendios/
├── FIRMS_CSVs/                      # Carpeta con CSVs históricos descargados de FIRMS
│   ├── modis_2000_Colombia.csv
│   ├── fire_archive_SV-C2_681505.csv
│   └── fire_nrt_J2V-C2_681504.csv
│
├── FIRMS_Colombia.csv               # Dataset combinado de Colombia (1.7M registros)
├── FIRMS_Cali.csv                   # Dataset filtrado de Cali (473 eventos)
├── dataset_incendios_ml.csv         # Dataset procesado para ML (492 registros)
│
├── incendios.py                     # Script inicial de descarga y visualización
├── construir_dataset_ml.py          # Script de construcción del dataset ML
├── entrenar_modelo.py               # Script de entrenamiento del modelo
├── predecir_con_sensores.py         # Script de predicción con sensores
│
├── modelo_incendios.pkl             # Modelo entrenado (generado)
├── modelo_metadata.pkl              # Metadata del modelo (generado)
├── modelo_evaluacion.png            # Gráficas de evaluación (generado)
│
├── requirements.txt                 # Dependencias Python
└── README.md                        # Este archivo
```

---

## 🔧 Instalación

### 1. Instalar dependencias

```powershell
pip install -r requirements.txt
```

**Dependencias principales:**
- `pandas` - Manipulación de datos
- `numpy` - Operaciones numéricas
- `matplotlib`, `seaborn` - Visualización
- `scikit-learn` - Machine Learning
- `joblib` - Serialización de modelos
- `requests` - Peticiones HTTP para APIs

---

## 📜 Descripción Detallada de Archivos

### 1️⃣ `incendios.py` (125 líneas)
**Propósito:** Cargar y visualizar datos históricos de incendios desde CSVs locales.

**Qué hace:**
- Lee todos los `.csv` de la carpeta `FIRMS_CSVs/`
- Normaliza columnas (`latitude`, `longitude`, `acq_date`)
- Filtra eventos de Cali usando BBOX (`-76.65,3.30,-76.45,3.55`)
- Guarda `FIRMS_Colombia.csv` y `FIRMS_Cali.csv`
- Genera gráfica de serie temporal mensual

**Cómo ejecutar:**
```powershell
python incendios.py
```

**Variables clave:**
```python
LOCAL_CSV_DIR = "FIRMS_CSVs"           # Carpeta con CSVs descargados
BBOX_CALI = "-76.65,3.30,-76.45,3.55"  # Bounding box de Cali
```

**Salida:**
- `FIRMS_Colombia.csv` - Todos los eventos de Colombia
- `FIRMS_Cali.csv` - Solo eventos en Cali (473 registros, 2012-2025)
- Gráfica comparativa mensual

---

### 2️⃣ `construir_dataset_ml.py` (269 líneas)
**Propósito:** Construir dataset balanceado para entrenamiento de ML.

**Qué hace:**
1. **Lee `FIRMS_Cali.csv`** (473 eventos confirmados)
2. **Crea grid espacial** (~500m) y agrupa por día/celda
3. **Genera muestras negativas** (5 por cada positivo) - días sin incendio
4. **Descarga datos meteorológicos** de Open-Meteo Archive API:
   - Temperatura (max/min/media)
   - Humedad relativa
   - Viento máximo
   - Precipitación (día + acumulada 7/14/30 días)
5. **Añade features temporales** (mes, día_año, estacion_seca)
6. **Exporta `dataset_incendios_ml.csv`**

**Cómo ejecutar:**
```powershell
python construir_dataset_ml.py
```

**Variables clave:**
```python
FIRMS_CALI_CSV = "FIRMS_Cali.csv"
OUTPUT_CSV = "dataset_incendios_ml.csv"
LAT_MIN, LAT_MAX = 3.30, 3.55         # Bbox Cali
LON_MIN, LON_MAX = -76.65, -76.45
GRID_SIZE = 0.005                      # ~500m en grados
NEGATIVE_RATIO = 5                     # 5 negativos por positivo
```

**Salida:**
- `dataset_incendios_ml.csv` (492 registros: 103 incendios + 389 no-incendios)

**Tiempo estimado:** 10-30 minutos (descarga meteorológica)

---

### 3️⃣ `entrenar_modelo.py` (231 líneas)
**Propósito:** Entrenar modelo de ML y evaluar performance.

**Qué hace:**
1. **Carga `dataset_incendios_ml.csv`**
2. **Selecciona features:**
   - **Sensores:** temperatura, humedad, viento, precipitación (7/14/30d)
   - **Temporales:** mes, día_año, estacion_seca
3. **Partición temporal** (NO aleatoria):
   - Entrenamiento: < 2023-01-01
   - Validación: >= 2023-01-01
4. **Entrena RandomForestClassifier**:
   - 100 árboles
   - `class_weight='balanced'` (para desbalance)
5. **Evalúa métricas**:
   - Precision, Recall, F1-score
   - Matriz de confusión
   - AUC-ROC, AUC-PR
6. **Guarda modelo** (`modelo_incendios.pkl`)
7. **Genera visualizaciones** (`modelo_evaluacion.png`)

**Cómo ejecutar:**
```powershell
python entrenar_modelo.py
```

**Variables clave:**
```python
DATASET_CSV = "dataset_incendios_ml.csv"
MODEL_FILE = "modelo_incendios.pkl"

SENSOR_FEATURES = [
    'temperatura_media',      # DHT11/DHT22
    'humedad_relativa',       # DHT11/DHT22
    'viento_max',            # Anemómetro
    'precipitacion_7d',      # Acumulada 7 días
    'precipitacion_14d',     # Acumulada 14 días
    'precipitacion_30d',     # Acumulada 30 días
]
```

**Salida:**
- `modelo_incendios.pkl` - Modelo entrenado
- `modelo_metadata.pkl` - Metadata (features, métricas, fecha)
- `modelo_evaluacion.png` - Gráficas (matriz confusión + importancia)

**Ejemplo de salida:**
```
REPORTE DE CLASIFICACIÓN:
              precision    recall  f1-score
No incendio       0.95      0.92      0.93
Incendio          0.78      0.85      0.81

AUC-ROC: 0.887
AUC-PR:  0.792
```

---

### 4️⃣ `predecir_con_sensores.py` (152 líneas)
**Propósito:** Hacer predicciones en tiempo real con datos de sensores.

**Qué hace:**
1. **Carga modelo entrenado** (`modelo_incendios.pkl`)
2. **Define datos de prueba hardcodeados** (líneas 29-44)
3. **Hace predicción** y muestra resultado
4. **Provee función reutilizable** `predecir_incendio()` (líneas 84-137)

**🔴 LÍNEAS HARDCODEADAS (29-44) - REEMPLAZAR CON SENSORES:**
```python
datos_sensores = {
    # 🔴 REEMPLAZAR con lecturas de DHT11/DHT22
    'temperatura_media': 32.5,      # Leer de sensor
    'humedad_relativa': 35.0,       # Leer de sensor
    
    # 🔴 REEMPLAZAR con lectura de anemómetro (o simular)
    'viento_max': 18.0,
    
    # 🔴 REEMPLAZAR con histórico o API meteorológica
    'precipitacion_7d': 0.0,
    'precipitacion_14d': 2.5,
    'precipitacion_30d': 8.0,
    
    # ✅ Estas se calculan automáticamente
    'mes': datetime.now().month,
    'dia_año': datetime.now().timetuple().tm_yday,
    'estacion_seca': 1 if datetime.now().month in [12,1,2,6,7,8] else 0
}
```

**Cómo ejecutar (modo prueba):**
```powershell
python predecir_con_sensores.py
```

**Función para integración con Wokwi (líneas 84-137):**
```python
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
```

**Ejemplo de uso con sensores reales:**
```python
# Leer sensores (ejemplo pseudocódigo Wokwi)
temperatura = dht11.temperature()
humedad = dht11.humidity()
viento = anemometro.read_speed()  # o simular
precip_7d = consultar_api_clima()  # o guardar histórico

# Predecir
resultado = predecir_incendio(temperatura, humedad, viento, 
                               precip_7d, precip_14d, precip_30d)

# resultado = {
#     'riesgo': True, 
#     'probabilidad': 0.85, 
#     'nivel': 'ALTO',
#     'timestamp': '2025-11-05T...'
# }

if resultado['riesgo']:
    activar_alarma()
    enviar_alerta()
```

---

## 🚀 Flujo Completo de Trabajo

### Fase 1: Obtención de Datos Históricos ✅ (COMPLETADO)
```powershell
# 1. Descargar CSVs de FIRMS y colocar en FIRMS_CSVs/
# 2. Procesar y visualizar
python incendios.py
# Salida: FIRMS_Cali.csv (473 eventos, 2012-2025)
```

### Fase 2: Construcción del Dataset ML ✅ (COMPLETADO)
```powershell
python construir_dataset_ml.py
# Salida: dataset_incendios_ml.csv (492 registros balanceados)
# Tiempo: ~20 minutos
```

### Fase 3: Entrenamiento del Modelo ✅ (COMPLETADO)
```powershell
python entrenar_modelo.py
# Salida: 
#   - modelo_incendios.pkl (modelo entrenado)
#   - modelo_evaluacion.png (gráficas)
#   - Métricas: AUC-ROC ~0.88, Precision ~0.78
```

### Fase 4: Integración con Sensores 🔄 (PENDIENTE - WOKWI)
```powershell
# Actualmente: prueba con datos simulados
python predecir_con_sensores.py

# Cuando tengas Wokwi:
# 1. Reemplazar líneas 29-44 con lecturas de sensores
# 2. Usar función predecir_incendio() con valores reales
# 3. Activar alertas según resultado
```

---

## 🔌 Integración con Sensores Wokwi

### Sensores necesarios (mencionaste):
- **DHT11/DHT22** → Temperatura y humedad ✅
- **MQ2** → Detección de gases/humo (opcional, para confirmar incendio activo)
- **Sensor de llama/IR** → Detección de fuego directo (opcional)
- **Anemómetro (simulado)** → Velocidad del viento

### Variables a reemplazar en `predecir_con_sensores.py`:

**ANTES (hardcodeado):**
```python
datos_sensores = {
    'temperatura_media': 32.5,       # 🔴 FIJO
    'humedad_relativa': 35.0,        # 🔴 FIJO
    'viento_max': 18.0,              # 🔴 FIJO
    'precipitacion_7d': 0.0,         # 🔴 FIJO
    # ...
}
```

**DESPUÉS (con Wokwi):**
```python
import dht_sensor  # Librería de tu simulación Wokwi

# Leer sensores reales
temperatura = dht_sensor.read_temperature()
humedad = dht_sensor.read_humidity()
viento = simular_viento()  # o leer de sensor
precip_7d = obtener_precipitacion_historica()  # API o base de datos

# Usar función de predicción
resultado = predecir_incendio(
    temperatura=temperatura,
    humedad=humedad,
    viento=viento,
    precip_7d=precip_7d,
    precip_14d=precip_14d,
    precip_30d=precip_30d
)

# Actuar según resultado
if resultado['riesgo'] and resultado['probabilidad'] > 0.7:
    print("🔴 ALERTA MÁXIMA")
    activar_sirena()
    enviar_notificacion()
elif resultado['riesgo']:
    print("🟠 PRECAUCIÓN")
    monitoreo_continuo()
else:
    print("✅ Sin riesgo")
```

---

## 📊 Datos del Dataset Final

### `dataset_incendios_ml.csv` (492 registros)
**Columnas (17 variables):**
1. `fecha` - Fecha del evento
2. `año` - Año (2012-2025)
3. `mes` - Mes (1-12)
4. `dia_año` - Día del año (1-365)
5. `estacion_seca` - Binario (1=seca, 0=lluvia)
6. `grid_lat` - Latitud de la celda
7. `grid_lon` - Longitud de la celda
8. `temperatura_max` - °C (máxima diaria)
9. `temperatura_min` - °C (mínima diaria)
10. `temperatura_media` - °C ← **SENSOR DHT11**
11. `humedad_relativa` - % ← **SENSOR DHT11**
12. `viento_max` - km/h ← **SENSOR ANEMÓMETRO**
13. `precipitacion_dia` - mm (día actual)
14. `precipitacion_7d` - mm (acumulada) ← **NECESARIO**
15. `precipitacion_14d` - mm (acumulada) ← **NECESARIO**
16. `precipitacion_30d` - mm (acumulada) ← **NECESARIO**
17. `incendio` - Etiqueta (0=no, 1=sí)

**Distribución:**
- 103 eventos positivos (incendios) - 20.9%
- 389 eventos negativos (no incendios) - 79.1%

---

## 🎯 Recomendaciones para Producción

### 1. Precipitación Acumulada
Para las variables `precipitacion_7d/14d/30d`:
- **Opción A:** Guardar histórico local (base de datos/CSV)
- **Opción B:** Consultar API gratuita (Open-Meteo, IDEAM)
- **Opción C:** Usar estación meteorológica local

### 2. Validación Espacial
Actualmente el modelo fue entrenado con datos de toda Cali. Para mejorar:
- Entrenar modelos por zona (Cristo Rey, Tres Cruces, etc.)
- Añadir features de elevación/pendiente

### 3. Re-entrenamiento
Actualizar modelo cada 6-12 meses con nuevos datos de FIRMS.

### 4. Umbrales de Alerta
Ajustar según necesidades:
```python
if probabilidad > 0.7:   # 70% → Alerta inmediata
if probabilidad > 0.5:   # 50% → Monitoreo intensivo
if probabilidad > 0.3:   # 30% → Precaución
```

---

## 📞 Soporte

Para consultas sobre:
- **Datos FIRMS:** https://firms.modaps.eosdis.nasa.gov/
- **Open-Meteo API:** https://open-meteo.com/
- **Scikit-learn:** https://scikit-learn.org/

---

## 📄 Licencia

Proyecto educativo - Sistema de predicción de incendios forestales.
Datos de FIRMS (NASA) - Uso libre para fines no comerciales.

---

## ✅ Checklist de Implementación

- [x] Descargar datos históricos FIRMS
- [x] Procesar y filtrar datos de Cali
- [x] Construir dataset balanceado con meteorología
- [x] Entrenar modelo RandomForest
- [x] Evaluar métricas (AUC-ROC: 0.88)
- [x] Crear función de predicción
- [ ] Integrar con sensores Wokwi
- [ ] Implementar sistema de alertas
- [ ] Desplegar en hardware (Arduino/ESP32)
- [ ] Pruebas en campo

---

**Última actualización:** 2025-11-05
**Estado:** ✅ Modelo entrenado y listo para integración con sensores

