# ======================================================
# 🔥 VISUALIZACIÓN DE INCENDIOS FIRMS DESDE CSVs LOCALES (Cali / Colombia)
# ======================================================
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN ---
LOCAL_CSV_DIR = "FIRMS_CSVs"   # carpeta con varios .csv históricos descargados
# Cali: zona urbana y cerros (lon_min, lat_min, lon_max, lat_max)
BBOX_CALI = "-76.65,3.30,-76.45,3.55"

# ======================================================
# 🔹 CARGA DE CSVs LOCALES
# ======================================================
def load_local_firms_csvs(directory: str) -> pd.DataFrame:
    """Lee todos los .csv del directorio y los concatena.
    Normaliza columnas clave: latitude, longitude, acq_date.
    """
    if not os.path.isdir(directory):
        print(f"❌ Carpeta no encontrada: {directory}")
        return pd.DataFrame()

    csv_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".csv")
    ]
    if not csv_files:
        print(f"❌ No se encontraron .csv en {directory}")
        return pd.DataFrame()

    frames = []
    for path in sorted(csv_files):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"⚠️ Error leyendo {path}: {e}")
            continue
        # Normalizar nombres
        df.columns = [c.strip().lower() for c in df.columns]
        rename_map = {}
        if "lat" in df.columns and "latitude" not in df.columns:
            rename_map["lat"] = "latitude"
        if "lon" in df.columns and "longitude" not in df.columns:
            rename_map["lon"] = "longitude"
        if "longitud" in df.columns and "longitude" not in df.columns:
            rename_map["longitud"] = "longitude"
        if "latitud" in df.columns and "latitude" not in df.columns:
            rename_map["latitud"] = "latitude"
        if "date" in df.columns and "acq_date" not in df.columns:
            rename_map["date"] = "acq_date"
        df = df.rename(columns=rename_map)

        # Columnas mínimas
        required = {"latitude", "longitude", "acq_date"}
        missing = required - set(df.columns)
        if missing:
            print(f"⚠️ {os.path.basename(path)} faltan columnas {missing}; se omite")
            continue
        df["source_file"] = os.path.basename(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ======================================================
# 🔹 PROCESAMIENTO
# ======================================================
fires_colombia = load_local_firms_csvs(LOCAL_CSV_DIR)
if not fires_colombia.empty:
    # Filtrar Cali por BBOX
    w, s, e, n = [float(x) for x in BBOX_CALI.split(",")]
    lon = fires_colombia["longitude"]
    lat = fires_colombia["latitude"]
    mask = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
    fires_cali = fires_colombia.loc[mask].copy()
    fires_cali["region"] = "Cali"
    fires_colombia["region"] = "Colombia"
else:
    fires_cali = pd.DataFrame()

# --- Guardar CSVs ---
if not fires_cali.empty:
    fires_cali.to_csv("FIRMS_Cali.csv", index=False)
if not fires_colombia.empty:
    fires_colombia.to_csv("FIRMS_Colombia.csv", index=False)

# ======================================================
# 🔹 VISUALIZACIÓN COMPARATIVA
# ======================================================
if not fires_cali.empty or not fires_colombia.empty:
    if not fires_cali.empty:
        fires_cali["acq_date"] = pd.to_datetime(fires_cali["acq_date"], errors="coerce")
    if not fires_colombia.empty:
        fires_colombia["acq_date"] = pd.to_datetime(fires_colombia["acq_date"], errors="coerce")

    series = []
    labels = []
    colors = []
    if not fires_cali.empty:
        cali_monthly = fires_cali.groupby(fires_cali["acq_date"].dt.to_period("M")).size()
        series.append(cali_monthly)
        labels.append("Cali")
        colors.append("orangered")
    if not fires_colombia.empty:
        col_monthly = fires_colombia.groupby(fires_colombia["acq_date"].dt.to_period("M")).size()
        series.append(col_monthly)
        labels.append("Colombia")
        colors.append("darkgreen")

    fig, ax = plt.subplots(figsize=(11, 6))
    for s, label, color in zip(series, labels, colors):
        s.sort_index().plot(ax=ax, label=label, color=color)
    ax.set_title("🔥 Detecciones FIRMS – Serie mensual")
    ax.set_ylabel("Número de incendios")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xlabel("Fecha")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No hay datos suficientes para graficar.")
