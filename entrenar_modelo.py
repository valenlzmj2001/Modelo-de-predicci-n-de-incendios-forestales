# ======================================================
# 🔥 ENTRENAMIENTO DE MODELO DE PREDICCIÓN DE INCENDIOS
# ======================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# --- CONFIGURACIÓN ---
DATASET_CSV = "dataset_incendios_ml.csv"
MODEL_FILE = "modelo_incendios.pkl"
SCALER_FILE = "scaler_incendios.pkl"

print("🔥 ENTRENAMIENTO DE MODELO DE PREDICCIÓN DE INCENDIOS")
print("=" * 60)

# ======================================================
# 1. CARGAR Y PREPARAR DATOS
# ======================================================
print("\n📂 Cargando dataset...")
df = pd.read_csv(DATASET_CSV)
df['fecha'] = pd.to_datetime(df['fecha'])

print(f"✅ Dataset cargado: {len(df)} registros")
print(f"   Incendios: {(df['incendio']==1).sum()} ({(df['incendio']==1).sum()/len(df)*100:.1f}%)")
print(f"   No incendios: {(df['incendio']==0).sum()} ({(df['incendio']==0).sum()/len(df)*100:.1f}%)")

# ======================================================
# 2. SELECCIONAR FEATURES (VARIABLES DE LOS SENSORES)
# ======================================================
# Estas son las variables que recibirás de tus sensores simulados
SENSOR_FEATURES = [
    'temperatura_media',      # DHT11/DHT22
    'humedad_relativa',       # DHT11/DHT22
    'viento_max',            # Anemómetro (simulado)
    'precipitacion_7d',      # Acumulada últimos 7 días
    'precipitacion_14d',     # Acumulada últimos 14 días
    'precipitacion_30d',     # Acumulada últimos 30 días
]

# Features temporales adicionales
TEMPORAL_FEATURES = [
    'mes',
    'dia_año',
    'estacion_seca'
]

ALL_FEATURES = SENSOR_FEATURES + TEMPORAL_FEATURES

print(f"\n📊 Variables de entrada (features):")
for feat in SENSOR_FEATURES:
    print(f"   🔹 {feat} ← sensor")
for feat in TEMPORAL_FEATURES:
    print(f"   📅 {feat} ← temporal")

X = df[ALL_FEATURES]
y = df['incendio']

# ======================================================
# 3. PARTICIÓN TEMPORAL (NO ALEATORIA)
# ======================================================
# Entrenar con datos antiguos, validar con recientes
print("\n📅 Partición temporal del dataset...")
split_date = '2023-01-01'
train_mask = df['fecha'] < split_date
test_mask = df['fecha'] >= split_date

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f"   Entrenamiento: {len(X_train)} registros (antes de {split_date})")
print(f"   Validación: {len(X_test)} registros (desde {split_date})")
print(f"   Balance entrenamiento: {(y_train==1).sum()} incendios / {(y_train==0).sum()} no-incendios")
print(f"   Balance validación: {(y_test==1).sum()} incendios / {(y_test==0).sum()} no-incendios")

# ======================================================
# 4. ENTRENAR MODELO
# ======================================================
print("\n🤖 Entrenando Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Importante para datos desbalanceados
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ Modelo entrenado")

# ======================================================
# 5. EVALUAR MODELO
# ======================================================
print("\n📊 Evaluando modelo en conjunto de validación...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Métricas
print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN:")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['No incendio', 'Incendio']))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("\nMATRIZ DE CONFUSIÓN:")
print(f"                 Predicho NO  Predicho SÍ")
print(f"Real NO          {cm[0,0]:>12} {cm[0,1]:>12}")
print(f"Real SÍ          {cm[1,0]:>12} {cm[1,1]:>12}")

# AUC-ROC y AUC-PR
if len(np.unique(y_test)) > 1:
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"\nAUC-ROC: {roc_auc:.3f}")
    print(f"AUC-PR:  {pr_auc:.3f}")

# ======================================================
# 6. IMPORTANCIA DE VARIABLES
# ======================================================
print("\n📊 Importancia de variables:")
feature_importance = pd.DataFrame({
    'feature': ALL_FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"   {row['feature']:.<30} {row['importance']:.4f}")

# ======================================================
# 7. GUARDAR MODELO
# ======================================================
print(f"\n💾 Guardando modelo entrenado...")
joblib.dump(model, MODEL_FILE)

# Guardar también metadata
metadata = {
    'fecha_entrenamiento': datetime.now().isoformat(),
    'features': ALL_FEATURES,
    'sensor_features': SENSOR_FEATURES,
    'n_train': len(X_train),
    'n_test': len(X_test),
    'roc_auc': roc_auc if len(np.unique(y_test)) > 1 else None,
    'pr_auc': pr_auc if len(np.unique(y_test)) > 1 else None
}
joblib.dump(metadata, 'modelo_metadata.pkl')

print(f"✅ Modelo guardado: {MODEL_FILE}")
print(f"✅ Metadata guardada: modelo_metadata.pkl")

# ======================================================
# 8. VISUALIZACIONES
# ======================================================
print("\n📊 Generando visualizaciones...")

# Matriz de confusión
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No incendio', 'Incendio'],
            yticklabels=['No incendio', 'Incendio'])
axes[0].set_title('Matriz de Confusión')
axes[0].set_ylabel('Real')
axes[0].set_xlabel('Predicho')

# Importancia de variables
feature_importance.plot(x='feature', y='importance', kind='barh', ax=axes[1], legend=False)
axes[1].set_title('Importancia de Variables')
axes[1].set_xlabel('Importancia')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('modelo_evaluacion.png', dpi=150)
print("✅ Gráficas guardadas: modelo_evaluacion.png")

# ======================================================
# 9. FUNCIÓN DE PREDICCIÓN SIMPLE (PARA USO CON SENSORES)
# ======================================================
print("\n" + "="*60)
print("🎯 FUNCIÓN DE PREDICCIÓN PARA SENSORES")
print("="*60)
print("""
Para hacer predicciones con tus sensores de Wokwi, usa:

    import joblib
    import pandas as pd
    
    # Cargar modelo
    modelo = joblib.load('modelo_incendios.pkl')
    
    # Datos de sensores (ejemplo)
    datos_sensores = {
        'temperatura_media': 28.5,      # °C (DHT11)
        'humedad_relativa': 45.0,       # % (DHT11)
        'viento_max': 15.0,             # km/h (simulado)
        'precipitacion_7d': 0.0,        # mm últimos 7 días
        'precipitacion_14d': 5.2,       # mm últimos 14 días
        'precipitacion_30d': 12.8,      # mm últimos 30 días
        'mes': 8,                       # agosto
        'dia_año': 220,                 # día 220 del año
        'estacion_seca': 1              # 1=seca, 0=lluvia
    }
    
    # Convertir a DataFrame
    df_sensor = pd.DataFrame([datos_sensores])
    
    # Predecir
    prediccion = modelo.predict(df_sensor)[0]
    probabilidad = modelo.predict_proba(df_sensor)[0, 1]
    
    print(f"Predicción: {'🔥 RIESGO DE INCENDIO' if prediccion == 1 else '✅ Sin riesgo'}")
    print(f"Probabilidad: {probabilidad*100:.1f}%")
""")

print("\n✅ Entrenamiento completado!")
print(f"\n📁 Archivos generados:")
print(f"   - {MODEL_FILE} (modelo entrenado)")
print(f"   - modelo_metadata.pkl (información del modelo)")
print(f"   - modelo_evaluacion.png (gráficas)")

