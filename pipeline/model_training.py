import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

import joblib
# =========================================================
# XGBoost
# =========================================================
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# =========================================================
# LOGGING
# =========================================================
LOG_PATH = Path("logs/pipeline.log")
LOG_PATH.parent.mkdir(exist_ok=True)


def write_log(level, msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} - {level} - {msg}\n"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


# =========================================================
# GRID DE HIPERPARÁMETROS XGB
# =========================================================
def build_xgb_param_grid():
    return {
        "model__n_estimators": [400, 600],
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0]
    }


# =========================================================
# ENTRENAMIENTO XGBOOST
# =========================================================
def run_xgb_training(preprocessor, X_train, y_train):
    if not HAS_XGB:
        raise RuntimeError("XGBoost no está instalado.")

    xgb_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror"
        ))
    ])

    param_grid = build_xgb_param_grid()
    write_log("INFO", f"Grid Search XGB: {param_grid}")

    search = GridSearchCV(
        xgb_pipe,
        param_grid,
        cv=5,
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
        refit="mse",
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    idx = search.best_index_
    best_mse = -search.cv_results_["mean_test_mse"][idx]
    best_r2 = search.cv_results_["mean_test_r2"][idx]

    write_log("INFO", f"Mejores hiperparámetros XGB: {search.best_params_}")
    write_log("INFO", f"CV MSE={best_mse:.4f}, CV R2={best_r2:.4f}")

    results_df = pd.DataFrame({
        "model": ["XGBoost"],
        "best_params": [search.best_params_],
        "mse": [best_mse],
        "r2": [best_r2]
    })

    return search.best_estimator_, results_df


# =========================================================
# GRAFICAR MÉTRICAS
# =========================================================
def plot_xgb_results(results_df):
    output_dir = Path("./Images/RESULTADOS")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        mse_val = results_df["mse"].iloc[0]
        r2_val = results_df["r2"].iloc[0]
    except Exception as e:
        write_log("ERROR", f"Error leyendo métricas para graficar: {e}")
        return

    plt.figure(figsize=(7, 5))
    sns.barplot(x=["MSE", "R2"], y=[mse_val, r2_val])
    plt.title("Desempeño del Modelo XGBoost")
    plt.ylabel("Valor")
    plt.tight_layout()

    out_path = output_dir / "xgb_metrics.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    write_log("INFO", f"Gráfico de métricas guardado en: {out_path}")


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================
def run_training_pipeline():

    write_log("INFO", "=== INICIO ENTRENAMIENTO XGBOOST ===")

    data_path = Path("./output/master_table.csv")

    try:
        df = pd.read_csv(data_path)
        write_log("INFO", f"Cargado master_table. Shape: {df.shape}")
    except Exception as e:
        write_log("ERROR", f"Error cargando CSV: {e}")
        return None, None, None
        
    df["last_review_date"] = df["last_review_date"].astype(str).str.strip()

    # Reemplazar valores no válidos conocidos por NaN
    df["last_review_date"] = df["last_review_date"].replace(
        ["", "nan", "None", "null", "Null", "UNKNOWN", "Unknown", "unknown", "UNK", "unk", "Unk"],
        pd.NA
    )
    # Convertir a fecha (todo lo que falle queda en NaT)
    df["last_review_date"] = pd.to_datetime(df["last_review_date"], errors="coerce")

    # Rellenar todo lo que sea inválido o NaT con la fecha requerida
    df["last_review_date"] = df["last_review_date"].fillna(pd.to_datetime("2018-07-15"))
    write_log("INFO","---------------- SPLIT TEMPORAL ----------------")    
    # ---------------- SPLIT TEMPORAL ----------------
    train_mask = (df["last_review_date"] >= "2016-10-01") & (df["last_review_date"] <= "2018-01-31")
    test_mask  = (df["last_review_date"] >= "2018-02-01") & (df["last_review_date"] <= "2018-06-30")
    pred_mask  = (df["last_review_date"] >= "2018-07-01") & (df["last_review_date"] <= "2018-08-31")

    df_train = df.loc[train_mask].copy()
    df_test  = df.loc[test_mask].copy()
    df_pred  = df.loc[pred_mask].copy()

    write_log("INFO", f"Train: {df_train.shape}, Test: {df_test.shape}, Predicción futura: {df_pred.shape}")

    # ---------------- FEATURE ENGINEERING ----------------
    df["last_review_year"] = df["last_review_date"].dt.year
    df["last_review_month"] = df["last_review_date"].dt.month

    cols_drop = [
        "last_review_date",
        "customer_id",
        "log_cust_review_mean",
        "cust_review_std",
        "cust_num_good_reviews",
        "cust_num_bad_reviews"
    ]

    df_train = df_train.drop(columns=[c for c in cols_drop if c in df_train.columns])
    df_test  = df_test.drop(columns=[c for c in cols_drop if c in df_test.columns])
    

    target_col = "cust_review_mean"

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    X_pred = df_pred.drop(columns=[target_col], errors="ignore")
    y_real_pred = df_pred[target_col] if target_col in df_pred.columns else None

    numeric_cols = X_train.select_dtypes(include=["number"]).columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    # ---------------- ENTRENAMIENTO ----------------
    best_model, results_df = run_xgb_training(preprocessor, X_train, y_train)

    # ---------------- GUARDAR GRÁFICO ----------------
    plot_xgb_results(results_df)

    # ---------------- MÉTRICAS DE TEST ----------------
    y_pred_test = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    write_log("INFO", f"TEST MSE={test_mse:.4f}")
    write_log("INFO", f"TEST R2={test_r2:.4f}")

    # ---------------- PREDICCIÓN JUL–AGO 2018 ----------------
    y_pred_future = best_model.predict(X_pred)

    comparison_df = pd.DataFrame({
        "fecha": df_pred["last_review_date"].dt.strftime("%Y-%m-%d"),
        "real": y_real_pred.values,
        "prediccion": y_pred_future
    })
    # ---------------- GUARDAR EXCEL DE PREDICCIONES ----------------
    output_pred_dir = Path("./output/PREDICCIONES")
    output_pred_dir.mkdir(parents=True, exist_ok=True)

    excel_path = output_pred_dir / "proyecciones_XGBoost.xlsx"

    comparison_df.to_excel(excel_path, index=False)
    write_log("INFO", f"Archivo Excel con predicciones guardado en: {excel_path}")

    write_log("INFO", "Comparación futura generada para 07–08/2018")

    # ---------------- GUARDAR MODELO ----------------
    model_path = "./output/best_model_XGBOOST.pkl"
    joblib.dump(best_model, model_path)
    write_log("INFO", f"Modelo XGBoost guardado en: {model_path}")

    write_log("INFO", "=== FIN PIPELINE XGBOOST ===")


    

    # Ruta del archivo Excel
    excel_path = Path("./output/PREDICCIONES/proyecciones_XGBoost.xlsx")

    # Cargar datos
    df = pd.read_excel(excel_path)

    # Asegurarse de que la columna 'fecha' es tipo datetime
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Crear gráfico de línea comparando 'real' vs 'prediccion'
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='fecha', y='real', data=df, label='Real', marker='o')
    sns.lineplot(x='fecha', y='prediccion', data=df, label='Predicción', marker='x')

    plt.title('Comparación Real vs Predicción del Modelo XGBoost')
    plt.xlabel('Fecha')
    plt.ylabel('Valor de la Revisión')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Guardar gráfico
    output_dir = Path("./Images/PREDICCIONES")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "comparacion_predicciones.png", dpi=300)

    return best_model, results_df, comparison_df


# =========================================================
# EJECUCIÓN DIRECTA
# =========================================================
if __name__ == "__main__":
    run_training_pipeline()
