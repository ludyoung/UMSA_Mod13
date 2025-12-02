import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pipeline.utils import log
import matplotlib.pyplot as plt
from pathlib import Path

def generar_features(dfs: dict) -> pd.DataFrame:
    log("=== INICIO Feature Engineering Avanzado ===")

    # -------------------------------
    # Cargar tablas base
    # -------------------------------
    log("Cargando tablas base...")
    orders = dfs["orders"]
    order_items = dfs["order_items"]
    reviews = dfs["reviews"]
    products = dfs["products"]
    sellers = dfs["sellers"]
    payments = dfs["payments"]
    customers = dfs["customers"]
    geo = dfs["geolocation"]
    cat_trans = dfs["category_translation"]

    # -------------------------------
    # Merge maestro
    # -------------------------------
    df = (
        orders.merge(order_items, on="order_id", how="left")
        .merge(reviews, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(sellers, on="seller_id", how="left")
        .merge(payments, on="order_id", how="left")
        .merge(customers, on="customer_id", how="left")
    )

    # -------------------------------
    # Fechas y retrasos
    # -------------------------------
    log("Procesando variables temporales y retrasos...")
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "review_creation_date",
        "review_answer_timestamp"
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Retrasos y tiempos
    df["delivery_time_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df["delay_vs_estimated"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
    df["approval_delay"] = (df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    df["carrier_delay"] = (df["order_delivered_carrier_date"] - df["order_approved_at"]).dt.total_seconds() / 3600
    df["was_late"] = (df["delay_vs_estimated"] > 0).astype(int)

    # Variables de compra
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_day"] = df["order_purchase_timestamp"].dt.day
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

    # Variables de review
    df["review_year"] = df["review_creation_date"].dt.year
    df["review_month"] = df["review_creation_date"].dt.month
    df["review_year_month"] = df["review_creation_date"].dt.to_period("M")
    df["review_quarter"] = df["review_creation_date"].dt.quarter
    df["review_semester"] = (df["review_creation_date"].dt.month - 1)//6 + 1

    # -------------------------------
    # Features por producto
    # -------------------------------
    log("Generando features por producto...")
    product_group = df.groupby("product_id").agg(
        prod_price_mean=("price", "mean"),
        prod_price_std=("price", "std"),
        prod_price_max=("price", "max"),
        prod_price_min=("price", "min"),
        prod_weight_mean=("product_weight_g", "mean"),
        prod_weight_std=("product_weight_g", "std"),
        prod_weight_max=("product_weight_g", "max"),
        prod_weight_min=("product_weight_g", "min"),
        prod_length_mean=("product_length_cm", "mean"),
        prod_height_mean=("product_height_cm", "mean"),
        prod_width_mean=("product_width_cm", "mean"),
        prod_volume_mean=("product_length_cm", lambda x: (x * df.loc[x.index, "product_height_cm"] * df.loc[x.index, "product_width_cm"]).mean() if "product_height_cm" in df and "product_width_cm" in df else 0),
        prod_volume_std=("product_length_cm", lambda x: (x * df.loc[x.index, "product_height_cm"] * df.loc[x.index, "product_width_cm"]).std() if "product_height_cm" in df and "product_width_cm" in df else 0),
        prod_is_heavy=("product_weight_g", lambda x: (x > x.mean()).mean()),
        prod_review_mean=("review_score", "mean"),
        prod_review_std=("review_score", "std"),
        prod_review_min=("review_score", "min"),
        prod_review_max=("review_score", "max"),
        prod_review_count=("review_score", "count"),
        prod_review_skew=("review_score", lambda x: x.skew()),
        prod_review_kurt=("review_score", lambda x: x.kurt())
    )
    product_group["prod_price_per_weight"] = product_group["prod_price_mean"] / (product_group["prod_weight_mean"] + 1e-6)
    product_group["prod_volume_per_weight"] = product_group["prod_volume_mean"] / (product_group["prod_weight_mean"] + 1e-6)
    product_group["log_prod_price"] = np.log1p(product_group["prod_price_mean"])
    product_group["log_prod_weight"] = np.log1p(product_group["prod_weight_mean"])
    product_group["log_prod_volume"] = np.log1p(product_group["prod_volume_mean"])
    df = df.merge(product_group, on="product_id", how="left")

    # -------------------------------
    # Features del vendedor
    # -------------------------------
    log("Generando features del vendedor...")
    seller_group = df.groupby("seller_id").agg(
        seller_num_products=("product_id", "nunique"),
        seller_total_orders=("order_id", "count"),
        seller_avg_price=("price", "mean"),
        seller_price_std=("price", "std"),
        seller_price_max=("price", "max"),
        seller_price_min=("price", "min"),
        seller_review_mean=("review_score", "mean"),
        seller_review_std=("review_score", "std"),
        seller_review_min=("review_score", "min"),
        seller_review_max=("review_score", "max"),
        seller_review_count=("review_score", "count"),
        seller_review_skew=("review_score", lambda x: x.skew()),
        seller_review_kurt=("review_score", lambda x: x.kurt()),
        seller_weight_mean=("product_weight_g", "mean"),
        seller_weight_std=("product_weight_g", "std"),
        seller_weight_max=("product_weight_g", "max"),
        seller_weight_min=("product_weight_g", "min"),
        seller_avg_delivery=("delivery_time_days", "mean"),
        seller_max_delivery=("delivery_time_days", "max"),
        seller_late_ratio=("was_late", "mean"),
        seller_volume_mean=("product_length_cm", lambda x: (x * df.loc[x.index, "product_height_cm"] * df.loc[x.index, "product_width_cm"]).mean() if "product_height_cm" in df and "product_width_cm" in df else 0),
        seller_volume_std=("product_length_cm", lambda x: (x * df.loc[x.index, "product_height_cm"] * df.loc[x.index, "product_width_cm"]).std() if "product_height_cm" in df and "product_width_cm" in df else 0)
    )
    seller_group["seller_price_per_product"] = seller_group["seller_avg_price"] / (seller_group["seller_num_products"] + 1e-6)
    seller_group["seller_volume_per_weight"] = seller_group["seller_volume_mean"] / (seller_group["seller_weight_mean"] + 1e-6)
    seller_group["log_seller_avg_price"] = np.log1p(seller_group["seller_avg_price"])
    seller_group["log_seller_weight_mean"] = np.log1p(seller_group["seller_weight_mean"])
    seller_group["log_seller_volume_mean"] = np.log1p(seller_group["seller_volume_mean"])
    df = df.merge(seller_group, on="seller_id", how="left")

    # -------------------------------
    # Features de pagos
    # -------------------------------
    log("Generando features de pagos...")
    payment_group = df.groupby("customer_id").agg(
        pay_total=("payment_value", "sum"),
        pay_mean=("payment_value", "mean"),
        pay_std=("payment_value", "std"),
        pay_min=("payment_value", "min"),
        pay_max=("payment_value", "max"),
        pay_median=("payment_value", "median"),
        pay_installments_mean=("payment_installments", "mean"),
        pay_installments_max=("payment_installments", "max"),
        pay_installments_std=("payment_installments", "std"),
        pay_num_methods=("payment_type", "nunique"),
        pay_count=("payment_value", "count")
    )
    for method in df["payment_type"].dropna().unique():
        payment_group[f"pay_ratio_{method}"] = (
            df[df["payment_type"] == method].groupby("customer_id")["payment_value"].count() /
            df.groupby("customer_id")["payment_value"].count()
        )
    df = df.merge(payment_group, on="customer_id", how="left")

    # -------------------------------
    # Features por cliente
    # -------------------------------
    log("Generando features por cliente y reviews...")
    customer_features = df.groupby("customer_id").agg(
        cust_total_orders=("order_id", "count"),
        cust_total_spent=("price", "sum"),
        cust_avg_price=("price", "mean"),
        cust_price_std=("price", "std"),
        cust_price_max=("price", "max"),
        cust_price_min=("price", "min"),
        cust_avg_freight=("freight_value", "mean"),
        cust_num_products=("product_id", "nunique"),
        cust_review_mean=("review_score", "mean"),
        cust_review_std=("review_score", "std"),
        cust_review_min=("review_score", "min"),
        cust_review_max=("review_score", "max"),
        cust_review_median=("review_score", "median"),
        cust_review_skew=("review_score", lambda x: x.skew()),
        cust_review_kurt=("review_score", lambda x: x.kurt()),
        cust_num_bad_reviews=("review_score", lambda x: (x <= 2).sum()),
        cust_num_good_reviews=("review_score", lambda x: (x >= 4).sum()),
        review_count=("review_score", "count"),
        cust_avg_delivery=("delivery_time_days", "mean"),
        cust_late_ratio=("was_late", "mean"),
        cust_avg_delay=("delay_vs_estimated", "mean"),
        cust_max_delay=("delay_vs_estimated", "max"),
        cust_min_delay=("delay_vs_estimated", "min"),
        cust_avg_month=("purchase_month", "mean"),
        cust_avg_weekday=("purchase_weekday", "mean"),
        cust_avg_hour=("purchase_hour", "mean"),
        cust_payment_total=("pay_total", "mean"),
        cust_payment_methods=("pay_num_methods", "mean"),
        cust_installments_mean=("pay_installments_mean", "mean"),
        cust_installments_max=("pay_installments_max", "max"),
        last_review_date=("review_creation_date", "max")
    ).reset_index()

    # -------------------------------
    # Features derivados adicionales
    # -------------------------------
    customer_features["ratio_good_bad_reviews"] = (
        customer_features["cust_num_good_reviews"] / (customer_features["cust_num_bad_reviews"] + 1)
    )
    customer_features["avg_spent_per_order"] = (
        customer_features["cust_total_spent"] / (customer_features["cust_total_orders"] + 1)
    )
    customer_features["log_total_spent"] = np.log1p(customer_features["cust_total_spent"])
    customer_features["log_avg_price"] = np.log1p(customer_features["cust_avg_price"])
    customer_features["log_avg_delivery"] = np.log1p(customer_features["cust_avg_delivery"])
    customer_features["review_range"] = customer_features["cust_review_max"] - customer_features["cust_review_min"]
    customer_features["review_ratio_extremes"] = (
        (customer_features["cust_num_bad_reviews"] + customer_features["cust_num_good_reviews"]) /
        (customer_features["review_count"] + 1)
    )

    # -------------------------------
    # Visualización jerárquica de features
    # -------------------------------
    def plot_feature_hierarchy(df_final, product_group, seller_group, payment_group):
        hierarchy = {
            "Producto": len(product_group.columns)+15,
            "Vendedor": len(seller_group.columns)+16,
            "Pagos": len(payment_group.columns)+13,
            "Cliente": {
                "Totales": len([c for c in df_final.columns if c.startswith("cust_total")]),
                "Reviews": len([c for c in df_final.columns if c.startswith("cust_review")]),
                "Entrega/Temporalidad": len([c for c in df_final.columns if c.startswith("cust_avg_delivery") or c.startswith("cust_late") or c.startswith("cust_avg_month")]),
                "Pagos derivados": len([c for c in df_final.columns if c.startswith("cust_payment") or c.startswith("cust_installments")]),
                "Polinomiales/Logs": len([c for c in df_final.columns if "log_" in c or "_x_" in c])
            },
            "Target": 1
        }
        categories = []
        counts = []
        colors = []
        base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for i, (key, value) in enumerate(hierarchy.items()):
            if isinstance(value, dict):
                total = sum(value.values())
                categories.append(key)
                counts.append(total)
                colors.append(base_colors[i % len(base_colors)])
            else:
                categories.append(key)
                counts.append(value)
                colors.append(base_colors[i % len(base_colors)])
        plt.figure(figsize=(14, 7))
        bars = plt.bar(categories, counts, color=colors)
        plt.title("Resumen Jerárquico de Features Generados")
        plt.ylabel("Número de Features")
        plt.xticks(rotation=30, ha="right")
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, yval, ha='center', va='bottom', fontsize=10)
        cliente_x = categories.index("Cliente")
        subcats = hierarchy["Cliente"]
        y_start = 0
        for subkey, subcount in subcats.items():
            y_mid = y_start + subcount / 2
            plt.text(cliente_x, y_mid, f"{subkey}: {subcount}", ha='center', va='center', fontsize=9, color='white')
            y_start += subcount
        output_dir = Path("./Images/FEATURES/")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "resumen_features_hierarchy.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Imagen jerárquica guardada en: {output_path}")
    # Antes de plot_feature_hierarchy
    total_rows = customer_features.shape[0]  # normalmente filas
    total_features = (
        len(customer_features.columns) +
        len(product_group.columns) +
        len(seller_group.columns) +
        len(payment_group.columns)
    )

    plot_feature_hierarchy(customer_features, product_group, seller_group, payment_group)

    
    log(f"Shape total combinando todas las tablas: filas={total_rows}, features totales={total_features}")
    log("=== FIN Feature Engineering Avanzado ===")
    return customer_features
