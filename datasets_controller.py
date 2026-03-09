import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
import warnings

# Suprimir advertencias de GLCM para imágenes muy homogéneas
warnings.filterwarnings("ignore")

class MicroDE_MultiOtsu:
    def __init__(self, NP=8, G_max=50, strategy='DE/rand/1'):
        """
        NP: Tamaño de la micro-población (mínimo 6 para rand/2).
        G_max: Número máximo de generaciones.
        strategy: Esquema de mutación.
        """
        self.NP = NP
        self.G_max = G_max
        self.strategy = strategy
        self.D = 3  # P0, P1, P2
        
    def _fitness(self, thresholds, hist, total_pixels):
        """Calcula la varianza negativa (para minimizar) de Multi-Otsu."""
        t = np.clip(np.sort(np.round(thresholds).astype(int)), 0, 255)
        
        # Probabilidades (pesos) y medias de las 4 clases
        w = np.zeros(4)
        mu = np.zeros(4)
        bins = np.arange(256)
        
        w[0] = np.sum(hist[:t[0]]) / total_pixels
        if w[0] > 0: mu[0] = np.sum(bins[:t[0]] * hist[:t[0]]) / (w[0] * total_pixels)
        
        w[1] = np.sum(hist[t[0]:t[1]]) / total_pixels
        if w[1] > 0: mu[1] = np.sum(bins[t[0]:t[1]] * hist[t[0]:t[1]]) / (w[1] * total_pixels)
        
        w[2] = np.sum(hist[t[1]:t[2]]) / total_pixels
        if w[2] > 0: mu[2] = np.sum(bins[t[1]:t[2]] * hist[t[1]:t[2]]) / (w[2] * total_pixels)
        
        w[3] = np.sum(hist[t[2]:]) / total_pixels
        if w[3] > 0: mu[3] = np.sum(bins[t[2]:] * hist[t[2]:]) / (w[3] * total_pixels)
        
        mu_t = np.sum(w * mu)
        sigma_b_sq = np.sum(w * ((mu - mu_t) ** 2))
        
        return -sigma_b_sq

    def optimize(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        total_pixels = image.size
        
        pop = np.random.uniform(0, 255, (self.NP, self.D))
        pop = np.sort(pop, axis=1)
        F = np.random.uniform(0.1, 1.0, self.NP)
        CR = np.random.uniform(0.1, 1.0, self.NP)
        
        fitness = np.array([self._fitness(ind, hist, total_pixels) for ind in pop])
        
        for g in range(self.G_max):
            best_idx = np.argmin(fitness)
            
            for i in range(self.NP):
                F_i = F[i] if np.random.rand() > 0.1 else np.random.uniform(0.1, 1.0)
                CR_i = CR[i] if np.random.rand() > 0.1 else np.random.uniform(0.1, 1.0)
                
                idxs = [idx for idx in range(self.NP) if idx != i]
                np.random.shuffle(idxs)
                r1, r2, r3, r4, r5 = idxs[:5]
                
                if self.strategy == 'DE/rand/1':
                    V = pop[r1] + F_i * (pop[r2] - pop[r3])
                elif self.strategy == 'DE/best/1':
                    V = pop[best_idx] + F_i * (pop[r1] - pop[r2])
                elif self.strategy == 'DE/rand/2':
                    V = pop[r1] + F_i * (pop[r2] - pop[r3] + pop[r4] - pop[r5])
                elif self.strategy == 'DE/best/2':
                    V = pop[best_idx] + F_i * (pop[r1] - pop[r2] + pop[r3] - pop[r4])
                
                V = np.clip(V, 0, 255)
                V = np.sort(V)
                
                # Crossover Binario
                j_rand = np.random.randint(self.D)
                mask = (np.random.rand(self.D) <= CR_i) | (np.arange(self.D) == j_rand)
                U = np.where(mask, V, pop[i])
                
                # Selección
                f_U = self._fitness(U, hist, total_pixels)
                if f_U <= fitness[i]:
                    pop[i] = U
                    fitness[i] = f_U
                    F[i] = F_i
                    CR[i] = CR_i
                    
        best_idx = np.argmin(fitness)
        return np.round(pop[best_idx]).astype(int)

def extract_radiomics(image, thresholds):
    P0, P1, P2 = thresholds
    mu_2 = np.mean(image)
    sigma_2 = np.std(image)
    
    # Segmentación: Tomamos la región más brillante (clase 4) delimitada por P2
    binary_mask = (image >= P2).astype(int)
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask)
    
    num_regiones = len(regions)
    areas = [r.area for r in regions]
    area_max = max(areas) if areas else 0
    area_total = sum(areas)
    
    # GLCM: Matriz de co-ocurrencia (distancia=1, ángulo=0)
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_contraste = graycoprops(glcm, 'contrast')[0, 0]
    glcm_energia = graycoprops(glcm, 'energy')[0, 0]
    glcm_homogeneidad = graycoprops(glcm, 'homogeneity')[0, 0]
    
    return [P0, P1, P2, mu_2, sigma_2, num_regiones, area_max, area_total, glcm_contraste, glcm_energia, glcm_homogeneidad]

def run_pipeline(dataset_path):
    estrategias = ['DE/rand/1', 'DE/best/1', 'DE/rand/2', 'DE/best/2']
    base_dir = Path(dataset_path)
    
    # Obtener todas las imágenes ignorando train/test/val
    all_images = list(base_dir.rglob("*.jpeg"))
    
    for estrategia in estrategias:
        print(f"\nProcesando con esquema: {estrategia}")
        optimizer = MicroDE_MultiOtsu(NP=8, G_max=30, strategy=estrategia)
        dataset_rows = []
        
        for idx, img_path in enumerate(all_images):
            # Determinar la clase basada en el nombre de la carpeta contenedora
            label_name = img_path.parent.name.upper()
            y_label = 1 if label_name == "PNEUMONIA" else 0
            
            # Leer imagen en escala de grises
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # 1. Optimizar umbrales
            best_thresholds = optimizer.optimize(img)
            
            # 2. Extraer Radiomics
            features = extract_radiomics(img, best_thresholds)
            
            # 3. Construir vector X, Y
            row = features + [y_label]
            dataset_rows.append(row)
            
            if (idx + 1) % 100 == 0:
                print(f"  -> Procesadas {idx + 1}/{len(all_images)} imágenes...")
                
        # Guardar CSV
        columns = ['P_0', 'P_1', 'P_2', 'mu_2', 'sigma_2', 'num_regiones', 'area_max', 'area_total', 'GLCM_contraste', 'GLCM_energia', 'GLCM_homogeneidad', 'clase']
        df = pd.DataFrame(dataset_rows, columns=columns)
        
        # Formatear el nombre del archivo para evitar caracteres inválidos en la ruta
        safe_name = estrategia.replace('/', '_')
        output_csv = f"radiomics_{safe_name}.csv"
        df.to_csv(output_csv, index=False)
        print(f"Dataset guardado: {output_csv}")

# --- Ejecución ---
ruta_dataset = 'dataset/chest_xray'
run_pipeline(ruta_dataset)