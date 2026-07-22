import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from skimage.filters import threshold_multiotsu
from joblib import Parallel, delayed
import warnings
import logging

warnings.filterwarnings("ignore")

# =====================================================================
# --- CONFIGURACIÓN DE RUTAS Y LOGS ---
# =====================================================================
BASE_OUT_DIR = Path("results/brain")
LOG_DIR = BASE_OUT_DIR / "logs"
CSV_DIR = BASE_OUT_DIR / "csv_caracteristicas_radiomicas_brain"
IMG_DIR = BASE_OUT_DIR / "images"
METRICS_DIR = BASE_OUT_DIR / "metrics"

for d in [LOG_DIR, CSV_DIR, IMG_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Configurar logging para escribir en consola y en archivo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "ejecucion_radiomica.log", mode='a'),
        logging.StreamHandler()
    ]
)

# --- FUNCIÓN DE FITNESS EXTERNA ---
def calculate_fitness(thresholds, hist, total_pixels):
    t = np.clip(np.sort(np.round(thresholds).astype(int)), 0, 255)
    D = len(t)
    w = np.zeros(D + 1)
    mu = np.zeros(D + 1)
    bins = np.arange(256)
    
    ranges = [0] + list(t) + [256]
    
    for i in range(D + 1):
        start, end = ranges[i], ranges[i+1]
        if start >= end: continue 
        
        w[i] = np.sum(hist[start:end]) / total_pixels
        if w[i] > 0:
            mu[i] = np.sum(bins[start:end] * hist[start:end]) / (w[i] * total_pixels)
            
    mu_t = np.sum(w * mu)
    sigma_b_sq = np.sum(w * ((mu - mu_t) ** 2))
    return -sigma_b_sq 

# --- CLASE DE OPTIMIZACIÓN 1: MICRO-DE (µSADE) ---
class uSADE_MultiOtsu:
    def __init__(self, D, NP=5, max_fes=3000, strategy='DE/rand/1', restart_iters=10, e=1, Fl=0.1, Fu=0.9, tau1=0.1, tau2=0.1):
        self.D = D                      
        self.NP = NP                    
        self.max_fes = max_fes 
        self.strategy = strategy
        self.restart_iters = restart_iters 
        self.e = e                      
        self.Fl = Fl                    
        self.Fu = Fu                    
        self.tau1 = tau1
        self.tau2 = tau2

    def optimize(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        total_pixels = image.size
        
        pop = np.sort(np.random.uniform(0, 255, (self.NP, self.D)), axis=1)
        fitness = np.array([calculate_fitness(ind, hist, total_pixels) for ind in pop])
        
        F = np.random.uniform(self.Fl, self.Fl + self.Fu, self.NP)
        Cr = np.random.rand(self.NP)
        
        fes = self.NP 
        t = 1 
        
        convergence_fitness = [np.min(fitness)]
        convergence_fes = [fes]
        
        while fes < self.max_fes:
            if t % self.restart_iters == 0:
                sort_idx = np.argsort(fitness)
                pop, fitness = pop[sort_idx], fitness[sort_idx]
                F, Cr = F[sort_idx], Cr[sort_idx]
                
                for i in range(self.e):
                    if np.random.rand() < self.tau1: F[i] = self.Fl + np.random.rand() * self.Fu
                    if np.random.rand() < self.tau2: Cr[i] = np.random.rand()
                        
                for i in range(self.e, self.NP):
                    if fes >= self.max_fes: break
                    pop[i] = np.sort(np.random.uniform(0, 255, self.D))
                    fitness[i] = calculate_fitness(pop[i], hist, total_pixels)
                    fes += 1
                    F[i] = np.random.uniform(self.Fl, self.Fl + self.Fu)
                    Cr[i] = np.random.rand()
            
            best_idx = np.argmin(fitness)
            
            for i in range(self.NP):
                if fes >= self.max_fes: break 
                
                idxs = [idx for idx in range(self.NP) if idx != i]
                np.random.shuffle(idxs)
                r1, r2, r3 = idxs[:3] 
                
                if self.strategy == 'DE/rand/1':
                    V = pop[r1] + F[i] * (pop[r2] - pop[r3])
                elif self.strategy == 'DE/best/1':
                    V = pop[best_idx] + F[i] * (pop[r1] - pop[r2])
                
                V = np.sort(np.clip(V, 0, 255))
                
                j_rand = np.random.randint(self.D)
                mask = (np.random.rand(self.D) <= Cr[i]) | (np.arange(self.D) == j_rand)
                U = np.where(mask, V, pop[i])
                
                U = np.sort(U) 
                
                f_U = calculate_fitness(U, hist, total_pixels)
                fes += 1 
                
                if f_U <= fitness[i]:
                    pop[i], fitness[i] = U, f_U
            t += 1
            convergence_fitness.append(np.min(fitness))
            convergence_fes.append(fes)
                
        best_idx = np.argmin(fitness)
        return np.round(pop[best_idx]).astype(int), fitness[best_idx], (convergence_fes, convergence_fitness)

# --- CLASE DE OPTIMIZACIÓN 2: STANDARD DE ---
class StandardDE_MultiOtsu:
    def __init__(self, D, NP=100, max_fes=3000, strategy='DE/rand/1', F=0.5, Cr=0.9):
        self.D = D
        self.NP = NP if NP is not None else max(10, 5 * D) 
        self.max_fes = max_fes
        self.strategy = strategy
        self.F = F
        self.Cr = Cr

    def optimize(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        total_pixels = image.size

        pop = np.sort(np.random.uniform(0, 255, (self.NP, self.D)), axis=1)
        fitness = np.array([calculate_fitness(ind, hist, total_pixels) for ind in pop])

        fes = self.NP
        convergence_fitness = [np.min(fitness)]
        convergence_fes = [fes]

        while fes < self.max_fes:
            best_idx = np.argmin(fitness)

            for i in range(self.NP):
                if fes >= self.max_fes: break

                idxs = [idx for idx in range(self.NP) if idx != i]
                np.random.shuffle(idxs)
                r1, r2, r3, r4, r5 = idxs[:5]

                if self.strategy == 'DE/rand/1':
                    V = pop[r1] + self.F * (pop[r2] - pop[r3])
                elif self.strategy == 'DE/best/1':
                    V = pop[best_idx] + self.F * (pop[r1] - pop[r2])

                V = np.sort(np.clip(V, 0, 255))

                j_rand = np.random.randint(self.D)
                mask = (np.random.rand(self.D) <= self.Cr) | (np.arange(self.D) == j_rand)
                U = np.where(mask, V, pop[i])

                f_U = calculate_fitness(U, hist, total_pixels)
                fes += 1

                if f_U <= fitness[i]:
                    pop[i] = U
                    fitness[i] = f_U

            convergence_fitness.append(np.min(fitness))
            convergence_fes.append(fes)

        best_idx = np.argmin(fitness)
        return np.round(pop[best_idx]).astype(int), fitness[best_idx], (convergence_fes, convergence_fitness)

# --- EXTRACCIÓN DE CARACTERÍSTICAS ---
def extract_features(image, thresholds):
    mask = (image >= thresholds[-1]).astype(np.uint8)
    regions = regionprops(label(mask))
    areas = [r.area for r in regions]
    
    glcm = graycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
    
    features = list(thresholds) + [
        np.mean(image), np.std(image),
        len(regions), max(areas) if areas else 0, sum(areas),
        graycoprops(glcm, 'contrast')[0,0], graycoprops(glcm, 'energy')[0,0], graycoprops(glcm, 'homogeneity')[0,0]
    ]
    return features

def extract_features_unsegmented(image):
    # Sin segmentación: toda la imagen es la ROI
    mask = np.ones_like(image, dtype=np.uint8)
    regions = regionprops(label(mask))
    areas = [r.area for r in regions]
    
    glcm = graycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
    
    features = [
        np.mean(image), np.std(image),
        len(regions), max(areas) if areas else 0, sum(areas),
        graycoprops(glcm, 'contrast')[0,0], graycoprops(glcm, 'energy')[0,0], graycoprops(glcm, 'homogeneity')[0,0]
    ]
    return features

# =====================================================================
# --- RECOLECTOR DE RUTAS CON ORDEN DETERMINISTA ---
# =====================================================================
def get_dataset_paths(dataset_folder_path):
    paths = []
    p = Path(dataset_folder_path)
    clases = {"Sano": 0, "Enfermo": 1}
    
    for clase_nombre, label_val in clases.items():
        carpeta_clase = p / clase_nombre
        if carpeta_clase.exists():
            for img in carpeta_clase.rglob("*"):
                if img.suffix.lower() in ['.png', '.jpg', '.jpeg'] and img.is_file():
                    paths.append((img, label_val))
    
    # IMPORTANTE: Ordenamos las rutas alfabéticamente para que el guardado 
    # intermitente siempre sepa exactamente dónde se quedó.
    paths.sort(key=lambda x: str(x[0]))
    return paths

# --- FUNCIÓN PARA PLOTEAR LA MEDIANA ---
def plot_median_result(prob_name, est, D, img_data):
    img = img_data['img']
    thresholds = img_data['thresholds']
    convergence = img_data['convergence']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    if len(thresholds) == 0:
        axes[1].imshow(img, cmap='gray')
        axes[1].set_title("Sin segmentar (Original)")
        axes[1].axis('off')
    else:
        seg = np.zeros_like(img)
        colors = np.linspace(0, 255, D + 1)
        
        seg[img < thresholds[0]] = colors[0]
        for i in range(D - 1):
            mask = (img >= thresholds[i]) & (img < thresholds[i+1])
            seg[mask] = colors[i+1]
        seg[img >= thresholds[-1]] = colors[-1]
        
        axes[1].imshow(seg, cmap='gray')
        axes[1].set_title(f"Segmentada (D={D})\nUmbrales: {thresholds}")
        axes[1].axis('off')
    
    if convergence:
        fes_list, fitness_list = convergence
        axes[2].plot(fes_list, fitness_list, color='blue')
        axes[2].set_title("Convergencia (Fitness)")
        axes[2].set_xlabel("Llamadas a Función (FEs)")
        axes[2].set_ylabel("Varianza Intra-clase")
    else:
        axes[2].text(0.5, 0.5, "Standard Otsu\nNo hay curva de convergencia", 
                     ha='center', va='center', fontsize=12)
        axes[2].axis('off')
        
    plt.suptitle(f"Mediana (Fitness) - {prob_name} - {est} (D={D})")
    plt.tight_layout()
    safe_est = est.replace('/', '_')
    plt.savefig(IMG_DIR / f"Mediana_{prob_name}_{safe_est}_D{D}.png", dpi=300)
    plt.close(fig)

# --- FUNCIÓN DE TAREA PARALELA (MODIFICADA PARA RESUME) ---
def procesar_combinacion(prob_name, image_list, est, D):
    if est == 'Standard_Otsu' and D >= 7:
        logging.warning(f"[SALTANDO] {prob_name} - {est} (D={D}). Búsqueda exhaustiva inviable.")
        return None 

    safe_est = est.replace('/', '_')
    output_dir = CSV_DIR / prob_name
    
    # Solución para el límite de 260 caracteres en Windows (WinError 206)
    abs_output_dir = os.path.abspath(str(output_dir))
    if os.name == 'nt' and not abs_output_dir.startswith('\\\\?\\'):
        abs_output_dir = '\\\\?\\' + abs_output_dir
        
    os.makedirs(abs_output_dir, exist_ok=True)
    filename = Path(abs_output_dir) / f"Radiomics_{safe_est}_D{D}.csv"

    # Lógica de Interrupción y Reanudación
    start_idx = 0
    if filename.exists():
        try:
            df_existente = pd.read_csv(filename)
            start_idx = len(df_existente)
            if start_idx >= len(image_list):
                logging.info(f"[COMPLETADO PREVIAMENTE] {prob_name} | {est} | D={D}")
                return {'Problema': prob_name, 'Estrategia': est, 'Umbrales_D': D, 'Tiempo_Segundos': 0, 'Nota': 'Ya existía'}
            else:
                logging.info(f"[REANUDANDO] {prob_name} | {est} | D={D} -> Desde imagen {start_idx}/{len(image_list)}")
        except pd.errors.EmptyDataError:
            logging.warning(f"Archivo {filename} vacío. Reiniciando extracción para esta configuración.")
            start_idx = 0
    else:
        logging.info(f"[INICIANDO] {prob_name} | {est} | D={D}")

    start_time = time.time()
    
    th_cols = [f'P{i}' for i in range(D)] if est != 'Original' else []
    cols = th_cols + ['mu','std','num_regiones','area_max','area_total','contrast','energy','homogeneity', 'label', 'best_fitness']
    
    tracking_data = []

    # Iterar solo sobre las imágenes restantes
    for idx, (img_path, lbl) in enumerate(image_list[start_idx:], start=start_idx):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        total_pixels = img.size
        
        convergence_curve = []
        if est == 'Original':
            best_fitness = 0
            thresholds = []
            features = extract_features_unsegmented(img)
        elif est == 'Standard_Otsu':
            try: 
                thresholds = threshold_multiotsu(img, classes=D+1)
                best_fitness = calculate_fitness(thresholds, hist, total_pixels)
            except: 
                continue
            features = extract_features(img, thresholds)
        elif est.startswith('uSADE'):
            strat_map = {'uSADE_rand_1': 'DE/rand/1', 'uSADE_best_1': 'DE/best/1'}
            optimizer = uSADE_MultiOtsu(D=D, strategy=strat_map[est], max_fes=D*1000)
            thresholds, best_fitness, convergence_curve = optimizer.optimize(img)
            features = extract_features(img, thresholds)
        elif est.startswith('DE'):
            strat_map = {'DE_rand_1': 'DE/rand/1', 'DE_best_1': 'DE/best/1'}
            optimizer = StandardDE_MultiOtsu(D=D, strategy=strat_map[est], max_fes=D*1000)
            thresholds, best_fitness, convergence_curve = optimizer.optimize(img)
            features = extract_features(img, thresholds)
        
        row_data = features + [lbl, best_fitness]
        
        # Guardado Fila por Fila (Append)
        df_row = pd.DataFrame([row_data], columns=cols)
        # Si el archivo no existe (o es la primera fila), escribe el encabezado
        write_header = not filename.exists()
        df_row.to_csv(filename, mode='a', header=write_header, index=False)
        
        tracking_data.append({
            'img': img, 'thresholds': thresholds,
            'convergence': convergence_curve, 'fitness': best_fitness
        })

        # Opcional: Descomenta esto si quieres ver el progreso en consola de cada imagen (puede llenar mucho la pantalla)
        # if idx % 50 == 0:
        #     logging.info(f"Progreso {prob_name} | {est} | D={D}: {idx}/{len(image_list)}")
    
    # Plotear la mediana solo si se generaron suficientes datos en esta corrida
    if tracking_data:
        tracking_data.sort(key=lambda x: x['fitness'])
        median_idx = len(tracking_data) // 2
        plot_median_result(prob_name, est, D, tracking_data[median_idx]) 
        
    elapsed_time = time.time() - start_time
    logging.info(f"<- [FINALIZADO PARCIAL/TOTAL] {prob_name} - {est} (D={D}) | Tiempo sesión: {round(elapsed_time, 2)}s")

    return {
        'Problema': prob_name, 'Estrategia': est, 'Umbrales_D': D,
        'Tiempo_Segundos': round(elapsed_time, 2), 'Tiempo_Minutos': round(elapsed_time / 60, 2)
    }

# --- PIPELINE MAESTRO ACTUALIZADO ---
def generate_all_datasets(main_path):
    logging.info("=== RECOPILANDO RUTAS DE IMÁGENES ===")
    problemas = {}
    
    base_dir = Path(main_path) / "Brain-X-Ray"
    
    if not base_dir.exists():
        logging.error(f"![ERROR] No se encontró la ruta esperada: {base_dir}")
        return

    for dataset_folder in base_dir.iterdir():
        if dataset_folder.is_dir():
            prob_name = dataset_folder.name
            rutas_encontradas = get_dataset_paths(dataset_folder)
            if rutas_encontradas:
                problemas[prob_name] = rutas_encontradas
                logging.info(f"[{prob_name}] Encontradas {len(rutas_encontradas)} imágenes.")
    
    estrategias = ['Original', 'uSADE_rand_1', 'uSADE_best_1', 'DE_rand_1', 'DE_best_1']
    umbrales = [3]
    
    lista_tareas = []
    for prob_name, image_list in problemas.items():
        for D in umbrales:
            for est in estrategias:
                if est == 'Original' and D != umbrales[0]:
                    continue
                D_val = 0 if est == 'Original' else D
                lista_tareas.append((prob_name, image_list, est, D_val))
            
    if not lista_tareas:
        logging.warning("No se generaron tareas. Revisa la estructura de tus carpetas.")
        return

    logging.info(f"\n=== INICIANDO EXTRACCIÓN PARALELA ({len(lista_tareas)} tareas) ===")
    logging.info("El script guardará el progreso fila por fila. Puedes detenerlo en cualquier momento.")
    
    timing_records = Parallel(n_jobs=-1)(
        delayed(procesar_combinacion)(prob, imgs, est, D) for prob, imgs, est, D in lista_tareas
    )

    timing_records = [r for r in timing_records if r is not None]

    logging.info("\n" + "="*50)
    logging.info("RESUMEN DE TIEMPOS DE PROCESAMIENTO (Sesión actual)")
    logging.info("="*50)
    df_times = pd.DataFrame(timing_records)
    
    # Evitar imprimir la tabla si está vacía
    if not df_times.empty:
        print(df_times.to_string(index=False))
        df_times.to_csv(METRICS_DIR / "Tiempos_Procesamiento_Sesion.csv", mode='a', header=not (METRICS_DIR / "Tiempos_Procesamiento_Sesion.csv").exists(), index=False)
        logging.info("[OK] Tabla de tiempos actualizada en 'results/brain/metrics/Tiempos_Procesamiento_Sesion.csv'")
    
    logging.info("[OK] Todos los resultados radiómicos están a salvo en 'results/brain/csv_caracteristicas_radiomicas_brain'.")

if __name__ == "__main__": 
    generate_all_datasets('dataset')