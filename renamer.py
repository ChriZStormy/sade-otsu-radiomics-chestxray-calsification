import os
import shutil
from pathlib import Path

# 1. Definir rutas basándonos en tu estructura actual y la deseada
DIR_ORIGEN = Path("results/brain/csv_caracteristicas_radiomicas_brain")
DIR_DESTINO = Path("results/radiomics-datasets-brain")

# 2. Mapeo para traducir los nombres largos de tus carpetas a los prefijos exactos que espera el nuevo script
mapeo_enfermedades = {
    "Alzheimer MRI Preprocessed Dataset_Binario": "Alzheimer",
    "Brain Stroke CT Dataset_Binario": "Stroke",
    "Brain Tumor MRI Dataset_Binario": "Tumor"
}

def renombrar_y_unificar_csvs():
    print("Iniciando la preparación de los datasets para el análisis estadístico...\n")
    
    # Crear la carpeta de destino automáticamente
    DIR_DESTINO.mkdir(parents=True, exist_ok=True)
    contador_archivos = 0

    if not DIR_ORIGEN.exists():
        print(f"⚠️ ERROR: No se encontró la ruta de origen: {DIR_ORIGEN}")
        return

    # 3. Iterar por cada enfermedad y sus archivos
    for nombre_carpeta, prefijo_enfermedad in mapeo_enfermedades.items():
        ruta_carpeta = DIR_ORIGEN / nombre_carpeta
        
        if not ruta_carpeta.exists():
            print(f"  - ⚠️ Carpeta no encontrada: {nombre_carpeta}")
            continue
            
        print(f"Procesando archivos de: {prefijo_enfermedad}")
        
        # Buscar los CSVs (Ej. Radiomics_DE_best_1_D3.csv)
        archivos_csv = list(ruta_carpeta.glob("Radiomics_*.csv"))
        
        for archivo_original in archivos_csv:
            # Reemplazar "Radiomics_" por "Radiomics_Enfermedad_"
            # Ej: Radiomics_DE_best_1_D3.csv -> Radiomics_Alzheimer_DE_best_1_D3.csv
            nuevo_nombre = archivo_original.name.replace("Radiomics_", f"Radiomics_{prefijo_enfermedad}_")
            ruta_destino = DIR_DESTINO / nuevo_nombre
            
            # Fix para el límite de caracteres en Windows (WinError 206)
            abs_src = os.path.abspath(str(archivo_original))
            abs_dst = os.path.abspath(str(ruta_destino))
            if os.name == 'nt':
                if not abs_src.startswith('\\\\?\\'): abs_src = '\\\\?\\' + abs_src
                if not abs_dst.startswith('\\\\?\\'): abs_dst = '\\\\?\\' + abs_dst
            
            # Usar copy2 es más seguro, así mantienes tu respaldo original intacto
            shutil.copy2(abs_src, abs_dst)
            contador_archivos += 1
            
        print(f"  -> {len(archivos_csv)} archivos renombrados y movidos exitosamente.")

    print(f"\n¡Completado! {contador_archivos} archivos listos en '{DIR_DESTINO}'.")
    print("Ya puedes ejecutar tu script de test de Friedman y Nemenyi.")

if __name__ == "__main__":
    renombrar_y_unificar_csvs()