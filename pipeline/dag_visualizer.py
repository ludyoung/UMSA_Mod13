# pipeline/dag_visualizer.py
from graphviz import Digraph

def generar_dag():
    """
    Genera el DAG del pipeline completo de procesamiento Olist.
    Siempre devuelve un objeto Graphviz válido o None si falla.
    """
    try:
        g = Digraph("Pipeline_Olist", format="png")
        g.attr(rankdir="LR", size="10,6")

        # ============================
        #          NODOS
        # ============================
        g.node("Load", "Carga de Datos")
        g.node("Validate", "Validación de Archivos")
        g.node("Clean", "Limpieza de Datos")
        g.node("Features", "Feature Engineering")
        g.node("Master", "Construcción de Master Table")
        g.node("EDA", "EDA sobre Master Table")
        g.node("Train", "Entrenamiento del Modelo")
        g.node("Eval", "Evaluación y Métricas")
        g.node("Export", "Exportación del Modelo (.pkl)")

        # ============================
        #         RELACIONES
        # ============================
        g.edge("Load", "Validate")
        g.edge("Validate", "Clean")
        g.edge("Clean", "Features")
        g.edge("Features", "Master")
        g.edge("Master", "EDA")
        g.edge("EDA", "Train")
        g.edge("Train", "Eval")
        g.edge("Eval", "Export")

        return g

    except Exception as e:
        print(f"Error generando DAG: {e}")
        return None
