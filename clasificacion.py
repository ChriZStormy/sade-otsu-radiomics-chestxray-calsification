import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from joblib import Parallel, delayed
import scipy.stats as stats
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# CREAR CARPETAS NECESARIAS AUTOMÁTICAMENTE
# ==========================================
# exist_ok=True evita que el programa marque error si la carpeta ya existe
os.makedirs("results/logs", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/images", exist_ok=True)

# ==========================================
# 0. CLASE PARA REDIRECCIONAR IMPRESIONES A TXT
# ==========================================
class Logger(object):
    def __init__(self, filename="results/logs/log_resultados_maestros.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("results/logs/log_resultados_maestros.txt")

# ==========================================
# 1. FUNCIONES DE PROCESAMIENTO
# ==========================================

def procesar_dataset(archivo, enfermedad, mutacion, D, n_splits=10, n_repeats=5):
    """
    Procesa el dataset. Captura la enfermedad, mutación y umbrales (D).
    """
    if not os.path.exists(archivo):
        return archivo, None, f"[AVISO] Archivo no encontrado (Omitiendo): {archivo}"

    df = pd.read_csv(archivo)
    
    if 'label' not in df.columns:
        return archivo, None, f"[ERROR] La columna 'label' no existe en: {archivo}"

    cols_umbrales = [col for col in df.columns if col.startswith('P') and col[1:].isdigit()]
    columnas_sesgo = cols_umbrales + ['best_fitness']
    cols_a_borrar = ['label'] + [c for c in columnas_sesgo if c in df.columns]
    
    X = df.drop(columns=cols_a_borrar, errors='ignore').values
    y = df['label'].values

    metricas = {
        'Enfermedad': [],
        'Mutacion': [],
        'Umbrales': [],
        'Iteracion': [],
        'acc_train': [], 
        'acc_test': [], 
        'recall_test': [], 
        'f1_test': [], 
        'auc_test': []
    }
    
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    fold = 1
    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        svm = SVC(kernel='linear', class_weight='balanced', random_state=42)
        svm.fit(X_train, y_train)

        y_pred_train = svm.predict(X_train)
        y_pred_test = svm.predict(X_test)
        y_scores_test = svm.decision_function(X_test)

        metricas['Enfermedad'].append(enfermedad)
        metricas['Mutacion'].append(mutacion)
        metricas['Umbrales'].append(f"D{D}") 
        metricas['Iteracion'].append(f"Fold_{fold}")
        
        metricas['acc_train'].append(accuracy_score(y_train, y_pred_train))
        metricas['acc_test'].append(accuracy_score(y_test, y_pred_test))
        metricas['recall_test'].append(recall_score(y_test, y_pred_test, zero_division=0))
        metricas['f1_test'].append(f1_score(y_test, y_pred_test, zero_division=0))
        metricas['auc_test'].append(roc_auc_score(y_test, y_scores_test))

        fold += 1

    df_metricas = pd.DataFrame(metricas)
    return archivo, df_metricas, None

# ==========================================
# 2. TEST ESTADÍSTICO MAESTRO (TODOS VS TODOS)
# ==========================================

def realizar_test_estadistico_maestro(df_completo, metrica='f1_test'):
    print(f"\n{'-'*80}")
    print(f"[ANÁLISIS ESTADÍSTICO MAESTRO: TODOS LOS ALGORITMOS VS TODOS LOS UMBRALES]")
    print(f"Métrica objetivo: {metrica}")
    print(f"{'-'*80}")

    # 1. Crear el tratamiento combinado (Ej. uSADE_rand_1_D12)
    df_completo['Estrategia_Global'] = df_completo['Mutacion'] + "_" + df_completo['Umbrales']
    
    # 2. El bloque ahora solo es la Enfermedad y el Fold
    df_completo['Bloque'] = df_completo['Enfermedad'] + "_" + df_completo['Iteracion']
    
    # Pivotar para que las columnas sean las 12 estrategias globales
    pivot_df = df_completo.pivot(index='Bloque', columns='Estrategia_Global', values=metrica)
    
    # Limpieza
    pivot_df = pivot_df.dropna(axis=1, how='all')
    pivot_df = pivot_df.dropna(axis=0, how='any')
    
    estrategias = pivot_df.columns.values 
    datos_por_estrategia = [pivot_df[est].values for est in estrategias]
    matriz_datos = pivot_df.values
    
    print(f"Total de bloques (Enfermedad x Fold) evaluados: {pivot_df.shape[0]}")
    print(f"Total de estrategias a competir: {len(estrategias)}\n")
    
    # ---------------------------------------------------------
    # TEST DE FRIEDMAN
    # ---------------------------------------------------------
    stat, p_value = stats.friedmanchisquare(*datos_por_estrategia)
    
    print(f"[TEST DE FRIEDMAN]")
    print(f"Estadístico Chi-cuadrado: {stat:.4f}")
    print(f"P-value: {p_value:.4e}")

    alfa = 0.05
    if p_value > alfa:
        print("\n[CONCLUSIÓN] No se rechaza H0. Todas las combinaciones rinden estadísticamente igual.")
        return None  
    
    print("\n[CONCLUSIÓN] Se rechaza H0. Hay diferencias significativas entre las configuraciones.")
    
    # ---------------------------------------------------------
    # RANKING MAESTRO (LA TABLA QUE PIDIÓ TU ASESOR)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"[RANKING DEFINITIVO - LAS 12 ESTRATEGIAS]")
    print("="*60)
    
    # Ranking: 1 es el mejor (mayor métrica)
    rangos_friedman = pivot_df.rank(axis=1, ascending=False).mean().sort_values()
    df_ranking = pd.DataFrame({'Ranking Promedio': rangos_friedman})
    df_ranking.index.name = 'Estrategia (Algoritmo + Umbral)'
    
    # Agregamos la media de F1-Score general para mayor claridad
    f1_means = pivot_df.mean().reindex(df_ranking.index)
    df_ranking['F1_Score_Medio'] = f1_means
    
    print(df_ranking.round(4).to_string())
    print("="*60 + "\n")
    
    ganador_absoluto = df_ranking.index[0]
    print(f"-> EL GANADOR ABSOLUTO DE LA INVESTIGACIÓN ES: {ganador_absoluto}")
    print("Procediendo con Post-Hoc Nemenyi...\n")

    # ---------------------------------------------------------
    # TEST POST-HOC DE NEMENYI (MATRIZ 12x12)
    # ---------------------------------------------------------
    p_values_nemenyi = sp.posthoc_nemenyi_friedman(matriz_datos)
    p_values_nemenyi.columns = estrategias
    p_values_nemenyi.index = estrategias
    
    # Gráfica del Heatmap Maestro (Más grande por ser 12x12)
    plt.figure(figsize=(14, 12)) 
    sns.heatmap(p_values_nemenyi, annot=True, cmap="YlGnBu", vmin=0, vmax=0.1, 
                cbar_kws={'label': 'P-value'}, fmt=".3f", linewidths=0.5, 
                annot_kws={"size": 8}) # Texto más pequeño para que quepa
    
    plt.title(f'Test Nemenyi Maestro (Todos vs Todos)\nMétrica: {metrica.upper()}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    archivo_grafica = f"results/images/heatmap_nemenyi_MAESTRO.png"
    plt.savefig(archivo_grafica, dpi=200) # Alta resolución para el documento
    plt.close()
    
    print(f"[ÉXITO] Mapa de calor 12x12 guardado como: {archivo_grafica}")
    return ganador_absoluto

# ==========================================
# 3. EJECUCIÓN PRINCIPAL (MAIN)
# ==========================================

if __name__ == "__main__":
    start_time = time.time()
    
    enfermedades = ["Alzheimer", "Stroke", "Tumor"]
    algoritmos = ['uSADE_rand_1', 'uSADE_best_1', 'DE_rand_1', 'DE_best_1']
    niveles_umbrales = [3, 6, 12]
    
    n_splits = 10
    n_repeats = 4
    
    print(f"============================================================")
    print(f"[INFO] Iniciando Pipeline de Consolidación Global")
    print(f"Validación Cruzada: {n_splits} splits x {n_repeats} repeticiones")
    print(f"============================================================")
    
    lista_tareas = []
    for D in niveles_umbrales:
        for enfermedad in enfermedades:
            for mutacion in algoritmos:
                archivo = f"results/radiomics-datasets-brain/Radiomics_{enfermedad}_{mutacion}_D{D}.csv"
                lista_tareas.append((archivo, enfermedad, mutacion, D))

    print(f"\n[Buscando {len(lista_tareas)} archivos CSV en paralelo...]")
    
    resultados = Parallel(n_jobs=-1)(
        delayed(procesar_dataset)(archivo, enf, mut, D, n_splits, n_repeats) 
        for archivo, enf, mut, D in lista_tareas
    )

    lista_dataframes = []
    for archivo, df_metricas, error in resultados:
        if error:
            print(error) 
        else:
            lista_dataframes.append(df_metricas)

    if lista_dataframes:
        df_global = pd.concat(lista_dataframes, ignore_index=True)
        
        # Guardar dataset crudo unificado
        df_global['Estrategia_Global'] = df_global['Mutacion'] + "_" + df_global['Umbrales']
        archivo_crudos = f"results/metrics/metricas_crudas_MAESTRA.csv"
        df_global.to_csv(archivo_crudos, index=False)
        print(f"\n[OK] Base de datos unificada guardada en: {archivo_crudos}")

        # --- TABLA RESUMEN POR ENFERMEDAD (F1-SCORE) ---
        print("\n" + "="*80)
        print(f"[TABLA MAESTRA DE RENDIMIENTO (F1-SCORE MEDIO)]")
        print("="*80)
        # Una vista súper limpia: Filas = Estrategia (Algoritmo+D), Columnas = Enfermedades
        tabla_resumen = df_global.groupby(['Estrategia_Global', 'Enfermedad'])['f1_test'].mean().unstack()
        tabla_resumen['Promedio_Total'] = tabla_resumen.mean(axis=1)
        tabla_resumen = tabla_resumen.sort_values(by='Promedio_Total', ascending=False)
        print(tabla_resumen.round(4).to_string())

        # --- PRUEBA ESTADÍSTICA MAESTRA (LOS 12 COMPETIDORES) ---
        ganador_absoluto = realizar_test_estadistico_maestro(df_global, metrica='f1_test')
            
    elapsed_time = time.time() - start_time
    print(f"\n[FIN] Todo el análisis se ha procesado en {elapsed_time/60:.2f} minutos.")
    print("[INFO] Revisa 'log_resultados_maestros.txt' y 'heatmap_nemenyi_MAESTRO.png'.")