# pipeline/feature_eng.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from pipeline.utils import log

# OUTPUT folder for artifacts
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


def _safe_period(x):
    """Devuelve Period('YYYY-MM') o NaN si no hay valor."""
    return x.to_period("M") if pd.notna(x) else pd.NaT


def _series_trend_monthly(series):
    """
    Calcula la pendiente (trend) de una serie ordenada por periodo (index asc).
    Usa np.polyfit si hay >=2 puntos, else 0.
    series: pandas Series indexed by period (o numeric index)
    """
    s = series.dropna()
    if len(s) < 2:
        return 0.0
    # x = 0..n-1
    x = np.arange(len(s))
    y = s.values.astype(float)
    try:
        coeff = np.polyfit(x, y, 1)
        return float(coeff[0])
    except Exception:
        return 0.0


def generar_features(dfs: dict, select_top_n: int = 30) -> pd.DataFrame:
    """
    Genera >100 features y selecciona los top-N features relacionados con cust_review_mean.
    Parámetros:
      - dfs: dict con las tablas necesarias (orders, order_items, reviews, products, sellers, payments, customers, geolocation)
      - select_top_n: número de features a seleccionar por importancia
    Retorna:
      - df_final: DataFrame con ['customer_id'] + top features + ['cust_review_mean']
    """

    log("=== INICIO Feature Engineering Avanzado (modo C) ===")

    # --------------------------
    # 0. Validaciones básicas
    # --------------------------
    required = [
        "orders",
        "order_items",
        "reviews",
        "products",
        "sellers",
        "payments",
        "customers",
        "geolocation",
    ]
    missing = [k for k in required if k not in dfs]
    if missing:
        raise KeyError(f"Faltan tablas en `dfs`: {missing}")

    # --------------------------
    # 1. Cargar tablas
    # --------------------------
    log("Cargando tablas base...")
    orders = dfs["orders"].copy()
    order_items = dfs["order_items"].copy()
    reviews = dfs["reviews"].copy()
    products = dfs["products"].copy()
    sellers = dfs["sellers"].copy()
    payments = dfs["payments"].copy()
    customers = dfs["customers"].copy()
    geolocation = dfs["geolocation"].copy()

    # Normalizar nombres de columnas si vienen con variantes
    # Geolocation suele venir con 'geolocation_zip_code_prefix' o 'zip_code_prefix'
    if "geolocation_zip_code_prefix" not in geolocation.columns and "zip_code_prefix" in geolocation.columns:
        geolocation = geolocation.rename(columns={"zip_code_prefix": "geolocation_zip_code_prefix"})
        log("Renombrada col 'zip_code_prefix' -> 'geolocation_zip_code_prefix' en geolocation")

    # --------------------------
    # 2. Pre-aggregar geolocation (evitar explosion)
    # --------------------------
    log("Agregando geolocation por zip prefix (evita explosion de joins)...")
    if "geolocation_zip_code_prefix" in geolocation.columns:
        geo_group = (
            geolocation.groupby("geolocation_zip_code_prefix")
            .agg(
                geo_lat_mean=("geolocation_lat", "mean"),
                geo_lng_mean=("geolocation_lng", "mean"),
                geo_count=("geolocation_lat", "count"),
            )
            .reset_index()
        )
        # merge geo -> customers por customer_zip_code_prefix si existe
        if "customer_zip_code_prefix" in customers.columns:
            customers = customers.merge(
                geo_group,
                left_on="customer_zip_code_prefix",
                right_on="geolocation_zip_code_prefix",
                how="left",
            )
            log("Merged geolocation aggregated into customers")
        else:
            log("[WARN] customers no tiene 'customer_zip_code_prefix'; se omite merge geolocation->customers")
    else:
        log("[WARN] geolocation no tiene geolocation_zip_code_prefix; se omite features geográficos")

    # --------------------------
    # 3. Merge controlado de tablas (orders + order_items) y joins por keys
    # --------------------------
    log("Construyendo dataframe base de pedidos (orders + order_items)...")
    # join orders + order_items -> order-level rows (uno por item)
    df = orders.merge(order_items, on="order_id", how="left", suffixes=("", "_oi"))

    # anexar reviews por order_id (puede contener múltiples reviews por order? en Olist es 1)
    if "order_id" in reviews.columns:
        df = df.merge(reviews, on="order_id", how="left", suffixes=("", "_rv"))

    # anexar payments por order_id (payments tiene filas por orden)
    if "order_id" in payments.columns:
        # payments puede tener varios payments por order; agregamos por order para no multiplicar filas
        pay_order = payments.groupby("order_id").agg(
            pay_order_total=("payment_value", "sum"),
            pay_order_installments=("payment_installments", "sum"),
            pay_order_methods=("payment_type", "nunique"),
        ).reset_index()
        df = df.merge(pay_order, on="order_id", how="left")

    # anexar products por product_id
    if "product_id" in products.columns and "product_id" in df.columns:
        df = df.merge(products, on="product_id", how="left", suffixes=("", "_prod"))

    # anexar sellers minimal info (seller_id)
    if "seller_id" in sellers.columns and "seller_id" in df.columns:
        sellers_min = sellers[["seller_id", "seller_city", "seller_state"]].drop_duplicates()
        df = df.merge(sellers_min, on="seller_id", how="left")

    # anexar customers (por customer_id)
    if "customer_id" in customers.columns and "customer_id" in df.columns:
        cust_min = customers.copy()
        # si customer ya tiene geo_group cols, vienen incluidos
        df = df.merge(cust_min, on="customer_id", how="left")

    log(f"Data merged shape (orders+items+...): {df.shape}")

    # --------------------------
    # 4. Fechas / temporales por pedido
    # --------------------------
    log("Procesando variables temporales por pedido...")
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # calcular tiempos y robustecer con coerce
    df["delivery_time_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df["delay_vs_estimated"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
    # convert to numeric hours (can be NaN)
    df["approval_delay_hours"] = (
        (df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    )
    df["carrier_delay_hours"] = (
        (df["order_delivered_carrier_date"] - df["order_approved_at"]).dt.total_seconds() / 3600
    )
    df["was_late"] = (df["delay_vs_estimated"] > 0).astype(int)

    # componentes temporales
    df["order_month"] = df["order_purchase_timestamp"].dt.month
    df["order_year"] = df["order_purchase_timestamp"].dt.year
    df["order_week"] = df["order_purchase_timestamp"].dt.isocalendar().week
    df["order_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df["order_hour"] = df["order_purchase_timestamp"].dt.hour

    # --------------------------
    # 5. Agregaciones por producto y seller (para merge posterior)
    # --------------------------
    log("Construyendo agregaciones por producto y seller...")
    prod_agg = df.groupby("product_id").agg(
        prod_orders=("order_id", "count"),
        prod_price_mean=("price", "mean"),
        prod_price_std=("price", "std"),
        prod_review_mean=("review_score", "mean"),
    ).reset_index()

    seller_agg = df.groupby("seller_id").agg(
        seller_orders=("order_id", "count"),
        seller_unique_products=("product_id", "nunique"),
        seller_price_mean=("price", "mean"),
        seller_review_mean=("review_score", "mean"),
    ).reset_index()

    # añadir agregados de producto/seller a df (opcional)
    df = df.merge(prod_agg, on="product_id", how="left", suffixes=("", "_pagg"))
    df = df.merge(seller_agg, on="seller_id", how="left", suffixes=("", "_sagg"))

    # --------------------------
    # 6. Construir customer_features base (asegura existencia antes de merges temporales)
    # --------------------------
    log("Generando customer_features base (RFM, reviews, delivery, pagos)...")
    cust_base = df.groupby("customer_id").agg(
        cust_total_orders=("order_id", "nunique"),
        cust_total_items=("order_item_id", "count"),
        cust_total_spent=("pay_order_total", "sum"),  # comes from aggregated payments
        cust_avg_price=("price", "mean"),
        cust_avg_freight=("freight_value", "mean"),
        cust_review_mean=("review_score", "mean"),
        cust_review_std=("review_score", "std"),
        cust_num_good_reviews=("review_score", lambda x: (x >= 4).sum()),
        cust_num_bad_reviews=("review_score", lambda x: (x <= 2).sum()),
        cust_avg_delivery=("delivery_time_days", "mean"),
        cust_late_ratio=("was_late", "mean"),
        cust_avg_approval_delay=("approval_delay_hours", "mean"),
        cust_avg_carrier_delay=("carrier_delay_hours", "mean"),
        cust_payment_installments_mean=("pay_order_installments", "mean"),
        cust_payment_methods=("pay_order_methods", "mean"),
        cust_unique_sellers=("seller_id", "nunique"),
        cust_unique_products=("product_id", "nunique"),
    ).reset_index()

    # Asegurar columnas numéricas y rellenar en cust_base mínimas
    cust_base = cust_base.astype({c: "float" for c in cust_base.columns if c != "customer_id"})
    cust_base.fillna(0, inplace=True)

    log(f"customer_features base shape: {cust_base.shape}")

    # --------------------------
    # 7. Features temporales por cliente (RFM, ventanas, tendencias)
    # --------------------------
    log("Generando features temporales (RFM, ventanas, tendencias)...")
    # recency
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    max_date = df["order_purchase_timestamp"].max()
    recency_ser = (
        df.groupby("customer_id")["order_purchase_timestamp"]
        .max()
        .apply(lambda d: (max_date - d).days if pd.notna(d) else np.nan)
        .rename("cust_recency_days")
    )

    # frequency (active months)
    df["year_month"] = df["order_purchase_timestamp"].dt.to_period("M")
    freq_month = df.groupby("customer_id")["year_month"].nunique().rename("cust_frequency_months")

    # monetary monthly (total spent / active months)
    def _monetary_month(g):
        months = g["year_month"].nunique()
        if months == 0:
            return 0.0
        return g["pay_order_total"].sum() / months

    monetary_month = df.groupby("customer_id").apply(_monetary_month).rename("cust_monetary_monthly")

    # counts in recent windows (3,6,12 months)
    if pd.isna(max_date):
        log("[WARN] max_date is NaN -> no temporal windows will be computed")
        orders_last_3m = pd.Series(dtype=float, name="cust_orders_last_3m")
        orders_last_6m = pd.Series(dtype=float, name="cust_orders_last_6m")
        orders_last_12m = pd.Series(dtype=float, name="cust_orders_last_12m")
    else:
        last_3m_period = (max_date.to_period("M") - 2)  # includes current month and -2 so window size 3
        last_6m_period = (max_date.to_period("M") - 5)
        last_12m_period = (max_date.to_period("M") - 11)

        orders_month = df.groupby(["customer_id", "year_month"])["order_id"].nunique().rename("orders_month_count").reset_index()

        orders_last_3m = (
            orders_month[orders_month["year_month"] >= last_3m_period]
            .groupby("customer_id")["orders_month_count"].sum()
            .rename("cust_orders_last_3m")
        )
        orders_last_6m = (
            orders_month[orders_month["year_month"] >= last_6m_period]
            .groupby("customer_id")["orders_month_count"].sum()
            .rename("cust_orders_last_6m")
        )
        orders_last_12m = (
            orders_month[orders_month["year_month"] >= last_12m_period]
            .groupby("customer_id")["orders_month_count"].sum()
            .rename("cust_orders_last_12m")
        )

    # trends (slope) for orders, reviews, delay by month
    log("Calculando tendencias mensuales (orders, review_score, delay) por cliente...")
    # orders per month series
    orders_month_s = orders_month.set_index(["customer_id", "year_month"])["orders_month_count"] if "orders_month" in locals() else None

    # compute monthly mean review and delay
    review_month = df.groupby(["customer_id", "year_month"])["review_score"].mean().rename("review_month_mean").reset_index()
    delay_month = df.groupby(["customer_id", "year_month"])["delay_vs_estimated"].mean().rename("delay_month_mean").reset_index()

    # helper to compute trend per customer from grouped month series
    def compute_group_trend(df_grouped, value_col, out_name):
        # df_grouped: DataFrame with customer_id, year_month, value_col
        out = (
            df_grouped.sort_values(["customer_id", "year_month"])
            .groupby("customer_id")[value_col]
            .apply(lambda s: _series_trend_monthly(s))
            .rename(out_name)
        )
        return out

    order_trend = compute_group_trend(orders_month.reset_index() if "orders_month" in locals() else orders_month, "orders_month_count", "cust_order_trend")
    review_trend = compute_group_trend(review_month, "review_month_mean", "cust_review_trend")
    delay_trend = compute_group_trend(delay_month, "delay_month_mean", "cust_delay_trend")

    # --------------------------
    # 8. Integrar todos los features en customer_features
    # --------------------------
    log("Integrando features temporales y base en customer_features final...")
    # Start from cust_base and merge the temporal series
    customer_features = cust_base.copy()

    to_merge = [
        recency_ser,
        freq_month,
        monetary_month,
        orders_last_3m,
        orders_last_6m,
        orders_last_12m,
        order_trend,
        review_trend,
        delay_trend,
    ]

    for s in to_merge:
        # s may be empty Series if not computed
        if s is None or len(s) == 0:
            continue
        customer_features = customer_features.merge(s.reset_index(), on="customer_id", how="left")

    # fill NaNs generated by merges
    customer_features.fillna(0, inplace=True)

    # --------------------------
    # 9. Features avanzados (interacciones, ratios, logs, booleans, counts)
    # --------------------------
    log("Generando features avanzados (interacciones, ratios, logs)...")
    # interactions (pairwise limited to a few to avoid explosion)
    interaction_pairs = [
        ("cust_total_spent", "cust_avg_delivery"),
        ("cust_total_spent", "cust_late_ratio"),
        ("cust_avg_price", "cust_avg_delivery"),
        ("cust_avg_price", "cust_late_ratio"),
        ("cust_payment_installments_mean", "cust_avg_price"),
    ]

    for a, b in interaction_pairs:
        col_name = f"{a}_x_{b}"
        customer_features[col_name] = customer_features.get(a, 0) * customer_features.get(b, 0)

    # ratios
    customer_features["ratio_good_bad_reviews"] = customer_features["cust_num_good_reviews"] / (
        customer_features["cust_num_bad_reviews"] + 1
    )

    # percentages
    customer_features["pct_bad_reviews"] = customer_features["cust_num_bad_reviews"] / (
        customer_features["cust_total_orders"] + 1
    )

    # logs
    for c in ["cust_total_spent", "cust_avg_delivery", "cust_review_mean", "cust_total_items"]:
        if c in customer_features.columns:
            customer_features[f"log_{c}"] = np.log1p(customer_features[c].astype(float).fillna(0))

    # boolean features example
    customer_features["has_bad_reviews"] = (customer_features["cust_num_bad_reviews"] > 0).astype(int)
    customer_features["high_spender_flag"] = (customer_features["cust_total_spent"] > customer_features["cust_total_spent"].median()).astype(int)

    # text-derived features - if review comments present in reviews table (word counts)
    if "review_comment_message" in df.columns:
        log("Generando features textuales: review comment lengths")
        comment_len = df.groupby("customer_id")["review_comment_message"].apply(lambda s: s.fillna("").str.len().sum()).rename("cust_total_comment_chars")
        customer_features = customer_features.merge(comment_len.reset_index(), on="customer_id", how="left")
        customer_features["cust_total_comment_chars"].fillna(0, inplace=True)

    # ensure numeric types
    for col in customer_features.columns:
        if col != "customer_id":
            customer_features[col] = pd.to_numeric(customer_features[col], errors="coerce").fillna(0)

    log(f"Total features finales antes de selección: {len(customer_features.columns)}")

    # --------------------------
    # 10. Selección automàtica (RandomForest)
    # --------------------------
    log("Ejecutando selección automática con RandomForest...")

    if "cust_review_mean" not in customer_features.columns:
        log("[WARN] cust_review_mean no está en customer_features; creando a partir de df (aggregate review_score)")
        agg_review = df.groupby("customer_id")["review_score"].mean().rename("cust_review_mean").reset_index()
        customer_features = customer_features.merge(agg_review, on="customer_id", how="left")

    target = customer_features["cust_review_mean"].copy()
    # politica: reemplazar NaN en target por 1 (según lo solicitado)
    nan_target = int(target.isna().sum())
    log(f"NaN en target cust_review_mean: {nan_target}")
    if nan_target > 0:
        log("Reemplazando NaN en target por 1")
        target = target.fillna(1)

    X = customer_features.drop(columns=["cust_review_mean", "customer_id"], errors="ignore").copy()

    # Convertir a numérico y limpiar inf
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # entrenar RF
    try:
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X, target)
    except Exception as e:
        log(f"[ERROR] Falló entrenamiento RandomForest: {e}")
        raise

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # guardar importancias
    fi_path = os.path.join(OUT_DIR, "feature_importances.csv")
    importances.to_csv(fi_path, header=["importance"])
    log(f"Feature importances guardadas en: {fi_path}")

    # log top 30
    top_n = min(select_top_n, len(importances))
    log(f"Top {top_n} variables más importantes:")
    for f, v in importances.head(top_n).items():
        log(f" - {f}: {v:.6f}")

    top_features = importances.head(top_n).index.tolist()

    # --------------------------
    # 11. Construir df_final (master table)
    # --------------------------
    cols_final = ["customer_id"] + [c for c in top_features if c in customer_features.columns] + ["cust_review_mean"]
    df_final = customer_features.loc[:, [c for c in cols_final if c in customer_features.columns]].copy()

    # garantizar orden y tipos
    df_final["customer_id"] = df_final["customer_id"].astype(str)
    numeric_cols = [c for c in df_final.columns if c != "customer_id"]
    df_final[numeric_cols] = df_final[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    out_master = os.path.join(OUT_DIR, "master_table.csv")
    df_final.to_csv(out_master, index=False)
    log(f"Master table creada y guardada: {out_master}")
    log(f"Shape final master_table: {df_final.shape}")

    log("=== FIN Feature Engineering Avanzado ===")
    return df_final
