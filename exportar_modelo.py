import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,ConfusionMatrixDisplay,roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import numpy as np

from src.database.connection import MySQLConnector


def main():
    print("=================================================================")
    print("  FASE 1: ENTRENAMIENTO Y PERSISTENCIA DEL MODELO (DEPLOYMENT)")
    print("=================================================================\n")

    db = MySQLConnector()
    query = """
        SELECT
            p.player_id,
            p.name AS jugador,
            p.position AS posicion,
            p.height_in_cm AS estatura_cm,
            TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
            p.market_value_in_eur AS valor_mercado
        FROM players p
        WHERE p.market_value_in_eur IS NOT NULL
          AND p.date_of_birth IS NOT NULL;
    """
    print("Extrayendo datos desde MySQL...")
    df = db.extract_data(query)

    # Preprocesamiento
    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['valor_mercado'] = pd.to_numeric(df['valor_mercado'], errors='coerce')
    df['posicion'] = df['posicion'].fillna('Sin Posicion')

    # Guardar las medianas de imputación para reutilizar en producción
    medianas_estatura = df.groupby('posicion')['estatura_cm'].median().to_dict()
    mediana_global_estatura = df['estatura_cm'].median()
    mediana_global_edad = df['edad'].median()

    df['estatura_cm'] = df.groupby('posicion')['estatura_cm'].transform(lambda x: x.fillna(x.median()))
    df['estatura_cm'] = df['estatura_cm'].fillna(mediana_global_estatura)
    df['edad'] = df['edad'].fillna(mediana_global_edad)

    umbral_elite = df['valor_mercado'].quantile(0.75)
    df['es_elite'] = (df['valor_mercado'] >= umbral_elite).astype(int)

    df_dummies = pd.get_dummies(df[['posicion']], drop_first=True, dtype=int)
    X = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)
    y = df['es_elite']

    print(f"[OK] Datos listos. Registros para entrenar: {len(X):,}")
    print(f"     Columnas del modelo: {X.columns.tolist()}\n")

    # -----------------------------------------------------------------------
    # EVALUACIÓN: DIVISIÓN ESTRATIFICADA Y DIAGNÓSTICO DE SOBREAJUSTE
    # -----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("[EVALUACIÓN] División estratificada 80/20")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    tree_libre = DecisionTreeClassifier(max_depth=None, random_state=42)
    tree_libre.fit(X_train, y_train)
    tree_podado = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    tree_podado.fit(X_train, y_train)

    acc_train_libre = accuracy_score(y_train, tree_libre.predict(X_train))
    acc_test_libre = accuracy_score(y_test, tree_libre.predict(X_test))
    acc_train_podado = accuracy_score(y_train, tree_podado.predict(X_train))
    acc_test_podado = accuracy_score(y_test, tree_podado.predict(X_test))
    print("[EVALUACIÓN] Diagnóstico de sobreajuste")
    print(f"  Árbol libre: Train={acc_train_libre:.4f}, Test={acc_test_libre:.4f}, Brecha={acc_train_libre-acc_test_libre:.4f}")
    print(f"  Árbol podado: Train={acc_train_podado:.4f}, Test={acc_test_podado:.4f}, Brecha={acc_train_podado-acc_test_podado:.4f}")

    # Optimización del árbol individual
    grid_arbol = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        {
            'max_depth': [2, 3, 4, 5, 6],
            'min_samples_leaf': [3, 5, 10, 15],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced', None]
        },
        cv=5, scoring='f1', n_jobs=-1
    )
    grid_arbol.fit(X_train, y_train)
    pred_arbol_optimo = grid_arbol.predict(X_test)
    print(f"  Árbol optimizado: {grid_arbol.best_params_}")
    print(f"  Accuracy Test={accuracy_score(y_test, pred_arbol_optimo):.4f}, F1 Élite={f1_score(y_test, pred_arbol_optimo):.4f}")

    # Panel original de diagnóstico: Accuracy, desbalance y matrices.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    etiquetas = ['Árbol sin límite', 'Árbol podado']
    x = np.arange(2)
    axes[0, 0].bar(x - 0.175, [acc_train_libre, acc_train_podado], 0.35, label='Train Accuracy')
    axes[0, 0].bar(x + 0.175, [acc_test_libre, acc_test_podado], 0.35, label='Test Accuracy')
    axes[0, 0].set_xticks(x, etiquetas)
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].set_title('Diagnóstico de sobreajuste y poda previa')
    axes[0, 0].legend()
    conteos = y.value_counts().reindex([0, 1], fill_value=0)
    axes[0, 1].bar(['Estándar (0)', 'Élite (1)'], conteos.values)
    axes[0, 1].set_title('Desbalance de clases')
    ConfusionMatrixDisplay(confusion_matrix(y_test, tree_libre.predict(X_test)),
                           display_labels=['Estándar', 'Élite']).plot(ax=axes[1, 0], colorbar=False)
    axes[1, 0].set_title('Matriz: árbol sin límite')
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred_arbol_optimo),
                           display_labels=['Estándar', 'Élite']).plot(ax=axes[1, 1], colorbar=False)
    axes[1, 1].set_title('Matriz: árbol optimizado')
    fig.tight_layout()
    fig.savefig('resultado_evaluacion_integral.png', dpi=300)
    plt.close(fig)

    # Comparación ROC-AUC e importancia Gini
    tree_roc = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
    tree_roc.fit(X_train, y_train)
    rf_roc = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=30,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_roc.fit(X_train, y_train)
    prob_tree = tree_roc.predict_proba(X_test)[:, 1]
    prob_rf = rf_roc.predict_proba(X_test)[:, 1]
    auc_tree = roc_auc_score(y_test, prob_tree)
    auc_rf = roc_auc_score(y_test, prob_rf)
    print(f"[EVALUACIÓN] ROC-AUC: Árbol={auc_tree:.4f} | Random Forest={auc_rf:.4f}")
    importancias = pd.Series(rf_roc.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("[EVALUACIÓN] Importancia de variables (Gini):")
    print(importancias.to_string())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fpr_tree, tpr_tree, _ = roc_curve(y_test, prob_tree)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)
    axes[0].plot(fpr_tree, tpr_tree, label=f'Árbol único (AUC={auc_tree:.3f})')
    axes[0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.3f})')
    axes[0].plot([0, 1], [0, 1], linestyle=':', label='Azar (AUC=0.500)')
    axes[0].set_xlabel('Tasa de falsos positivos (FPR)')
    axes[0].set_ylabel('Tasa de verdaderos positivos (TPR)')
    axes[0].set_title('Comparación de curvas ROC')
    axes[0].legend()
    importancias.head(6).sort_values().plot.barh(ax=axes[1])
    axes[1].set_xlabel('Importancia relativa (Gini)')
    axes[1].set_title('Importancia de variables del Random Forest')
    fig.tight_layout()
    fig.savefig('evaluacion_rf_vs_tree.png', dpi=300)
    plt.close(fig)
    print("[OK] Gráficos guardados: resultado_evaluacion_integral.png y evaluacion_rf_vs_tree.png")
    print("[NOTA] Los experimentos anteriores usan Train/Test; el modelo de despliegue se entrena después con todos los datos.")

    # -----------------------------------------------------------------------
    # TRATAMIENTO DE DESBALANCE Y GRIDSEARCHCV
    # -----------------------------------------------------------------------
    print("Optimización Sistemática con GridSearchCV y Penalización de Costo...")
    
    param_grid = {
        'max_depth': [4, 6, 8],
        'min_samples_leaf': [10, 30],
        'max_features': ['sqrt', None]
    }

    grid_search = GridSearchCV(
        estimator= RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        param_grid= param_grid,
        cv= 5,
        scoring= 'f1',
        n_jobs=-1
    )

    grid_search.fit(X, y)
    modelo_optimo = grid_search.best_estimator_
    print(f"  • Mejores Hiperparámetros: {grid_search.best_params_}")
    print(f"  • promedio del F1 en validación cruzada: {grid_search.best_score_:.4f}\n")

    # Crear paquete de despliegue
    paquete_despliegue = {
        'modelo': modelo_optimo,
        'columnas_X': X.columns.tolist(),
        'medianas_estatura': medianas_estatura,
        'mediana_global_estatura': mediana_global_estatura,
        'mediana_global_edad': mediana_global_edad,
        'umbral_elite_eur': umbral_elite
    }

    os.makedirs('models', exist_ok=True)
    ruta_modelo = 'models/modelo_scouting_rf.joblib'
    joblib.dump(paquete_despliegue, ruta_modelo)

    print(f"[OK] Paquete de modelo exportado exitosamente en: '{ruta_modelo}'")
    print(f"     Umbral de Élite aprendido: €{umbral_elite:,.0f}")

    
if __name__ == "__main__":
    main()