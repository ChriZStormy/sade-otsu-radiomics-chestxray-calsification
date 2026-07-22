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

warnings.filterwarnings("ignore")

# --- FUNCIÓN DE FITNESS EXTERNA ---
def calculate_fitness(thresholds, hist, total_pixels):
    t = np.clip(np.sort(np.round(thresholds).astype(int)), 0, 255)
    D = len(t)
    w = np.zeros(D + 1)
    mu = np.zeros(D + 1)
    bins = np.arange(256)
    
    # Crear rangos dinámicos basados en los umbrales
    ranges = [0] + list(t) + [256]
    
    for i in range(D + 1):
        start, end = ranges[i], ranges[i+1]
        if start >= end: continue # Prevenir errores si los umbrales colapsan
        
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
                
                # ... código anterior ...
                idxs = [idx for idx in range(self.NP) if idx != i]
                np.random.shuffle(idxs)
                
                r1, r2, r3 = idxs[:3] 
                
                if self.strategy == 'DE/rand/1':
                    V = pop[r1] + F[i] * (pop[r2] - pop[r3])
                elif self.strategy == 'DE/best/1':
                    V = pop[best_idx] + F[i] * (pop[r1] - pop[r2])
                # ... código siguiente ...
                
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
        # Población estándar suele ser 5 o 10 veces la dimensionalidad
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
    # Usamos el umbral más alto para aislar la ROI más brillante como referencia
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


# --- RECOLECTORES DE RUTAS ---
def     get_covid_paths(base):
    paths = []
    p = Path(base) / "Torax-X-Ray" / "Chest-X-Ray-COVID19"
    for folder, label_val in [("Normal", 0), ("COVID", 1)]:
        target_dir = p / folder / "images"
        if target_dir.exists():
            for img in target_dir.glob("*"):
                if img.suffix.lower() in ['.png', '.jpg', '.jpeg'] and "mask" not in str(img).lower():
                    paths.append((img, label_val))
    return paths

def get_neumonia_paths(base):
    paths = []
    p = Path(base) / "Torax-X-Ray" / "Chest-X-Ray-Neumonia"
    for img in p.rglob("*"):
        if img.suffix.lower() in ['.jpeg', '.jpg'] and "mask" not in str(img).lower():
            if img.is_file():
                if "NORMAL" in str(img).upper(): paths.append((img, 0))
                elif "PNEUMONIA" in str(img).upper(): paths.append((img, 1))
    return paths

def get_tb_paths(base):
    paths = []
    p = Path(base) / "Torax-X-Ray" / "Chest-X-Ray-Tuberculosis"
    csv_path = p / "MetaData.csv" 
    if not csv_path.exists(): return []

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        img_id = str(row['id'])
        label_val = int(row['ptb']) 
        img_name = f"{img_id}.png" 
        img_path = p / "Chest-X-Ray" / "image" / img_name
        
        if not img_path.exists():
            img_path = p / "images" / img_name

        if img_path.exists() and "mask" not in str(img_path).lower():
            paths.append((img_path, label_val))
            
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
        
        # Asignar colores dinámicamente según la cantidad de umbrales
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
    plt.savefig(f"results/images/Mediana_{prob_name}_{safe_est}_D{D}.png", dpi=300)
    plt.close(fig)

# --- FUNCIÓN DE TAREA PARALELA ---
# --- FUNCIÓN DE TAREA PARALELA ---
def procesar_combinacion(prob_name, image_list, est, D):
    if est == 'Standard_Otsu' and D >= 7:
        print(f" -> [SALTANDO] {prob_name} - {est} (D={D}). Búsqueda exhaustiva inviable O(L^{D}).")
        return None 

    # 1. Definir el nombre del archivo desde el principio
    safe_est = est.replace('/', '_')
    rel_filename = f"results/datasets/Radiomics_{prob_name}_{safe_est}_D{D}.csv"
    
    # Solución para el límite de 260 caracteres en Windows
    abs_filename = os.path.abspath(rel_filename)
    if os.name == 'nt' and not abs_filename.startswith('\\\\?\\'):
        abs_filename = '\\\\?\\' + abs_filename
        
    filename = abs_filename

    # 2. CONTROL DE ARCHIVOS EXISTENTES
    if os.path.exists(filename):
        print(f" -> [OMITIENDO] {filename} ya existe. Saltando procesamiento.")
        return {
            'Problema': prob_name,
            'Estrategia': est,
            'Umbrales_D': D,
            'Tiempo_Segundos': 0,
            'Tiempo_Minutos': 0,
            'Nota': 'Ya existia'
        }

    print(f" -> [INICIANDO] {prob_name} | {est} | D={D}")
    start_time = time.time()
    
    th_cols = [f'P{i}' for i in range(D)] if est != 'Original' else []
    cols = th_cols + ['mu','std','num_regiones','area_max','area_total','contrast','energy','homogeneity', 'label', 'best_fitness']
    
    data_rows = []
    tracking_data = []

    for img_path, lbl in image_list:
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
        data_rows.append(row_data)
        
        tracking_data.append({
            'img': img,
            'thresholds': thresholds,
            'convergence': convergence_curve,
            'fitness': best_fitness
        })
    
    if not data_rows:
        return None
        
    df_result = pd.DataFrame(data_rows, columns=cols)
    # 3. Guardar usando el nombre que definimos arriba
    df_result.to_csv(filename, index=False)
    
    if tracking_data:
        tracking_data.sort(key=lambda x: x['fitness'])
        median_idx = len(tracking_data) // 2
        median_item = tracking_data[median_idx]
        plot_median_result(prob_name, est, D, median_item) 
        
    elapsed_time = time.time() - start_time
    print(f" <- [FINALIZADO] {prob_name} - {est} (D={D}) | Tiempo: {round(elapsed_time, 2)}s")

    return {
        'Problema': prob_name,
        'Estrategia': est,
        'Umbrales_D': D,
        'Tiempo_Segundos': round(elapsed_time, 2),
        'Tiempo_Minutos': round(elapsed_time / 60, 2)
    }

# --- PIPELINE MAESTRO ---
def generate_all_datasets(main_path):
    os.makedirs('results/datasets', exist_ok=True)
    os.makedirs('results/images', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    print("=== RECOPILANDO RUTAS DE IMÁGENES ===")
    problemas = {
        "COVID19": get_covid_paths(main_path),
        "Neumonia": get_neumonia_paths(main_path),
        "Tuberculosis": get_tb_paths(main_path)
    }
    
    estrategias = ['Original', 'uSADE_rand_1', 'uSADE_best_1', 'DE_rand_1', 'DE_best_1']
    umbrales = [3]
    
    lista_tareas = []
    for prob_name, image_list in problemas.items():
        if not image_list:
            print(f"![AVISO] No hay imágenes para {prob_name}. Saltando...")
            continue
        print(f"[{prob_name}] Encontradas {len(image_list)} imágenes.")
        
        for D in umbrales:
            for est in estrategias:
                if est == 'Original' and D != umbrales[0]:
                    continue # Evitar procesar 'Original' varias veces si hubiera más umbrales en el futuro
                D_val = 0 if est == 'Original' else D
                lista_tareas.append((prob_name, image_list, est, D_val))
            
    if not lista_tareas:
        print("No se encontraron imágenes en las rutas especificadas. Revisa tu carpeta 'dataset'.")
        return

    print(f"\n=== INICIANDO EXTRACCIÓN PARALELA ({len(lista_tareas)} tareas) ===")
    print("Utilizando todos los núcleos disponibles. Esto puede tardar unos minutos...\n")
    
    timing_records = Parallel(n_jobs=-1)(
        delayed(procesar_combinacion)(prob, imgs, est, D) for prob, imgs, est, D in lista_tareas
    )

    # Filtrar los None devueltos por las ejecuciones omitidas
    timing_records = [r for r in timing_records if r is not None]

    print("\n" + "="*50)
    print("RESUMEN DE TIEMPOS DE PROCESAMIENTO")
    print("="*50)
    df_times = pd.DataFrame(timing_records)
    print(df_times.to_string(index=False))
    df_times.to_csv("results/metrics/Tiempos_Procesamiento_Totales.csv", index=False)
    print("\n[OK] Tabla de tiempos guardada como 'results/metrics/Tiempos_Procesamiento_Totales.csv'")
    print("[OK] Todos los CSVs y Gráficos PNG han sido guardados.")

if __name__ == "__main__": 
    generate_all_datasets('dataset')