import os

# Ruta base mostrada en tu imagen
base_path = os.path.join("dataset", "Brain-X-Ray")

# Nombres exactos de las carpetas de los datasets
datasets = [
    ("Alzheimer MRI Preprocessed Dataset_Binario", "dataset_alzheimer", "Alzheimer"),
    ("Brain Stroke CT Dataset_Binario", "dataset_stroke", "Stroke"),
    ("Brain Tumor MRI Dataset_Binario", "dataset_braintumor", "Tumor")
]

print("Resultados para pegar en tu artículo:\n")

for folder_name, cite_key, disease_name in datasets:
    dataset_path = os.path.join(base_path, folder_name)
    
    path_sano = os.path.join(dataset_path, "Sano")
    path_enfermo = os.path.join(dataset_path, "Enfermo")
    
    # Conteo de archivos (ignorando directorios en caso de haberlos)
    try:
        count_sano = len([f for f in os.listdir(path_sano) if os.path.isfile(os.path.join(path_sano, f))])
    except FileNotFoundError:
        count_sano = 0
        
    try:
        count_enfermo = len([f for f in os.listdir(path_enfermo) if os.path.isfile(os.path.join(path_enfermo, f))])
    except FileNotFoundError:
        count_enfermo = 0
        
    total = count_sano + count_enfermo
    
    # Imprime el formato LaTeX listo para usar
    print(f"    \\item \\textbf{{{folder_name.replace('_Binario', '')} \\cite{{{cite_key}}}}}:")
    print(f"    \\begin{{itemize}}")
    print(f"        \\item Class distribution (Total: {total:,} images):")
    print(f"        \\item Class 0 (Normal): {count_sano:,} images of healthy brains.")
    print(f"        \\item Class 1 ({disease_name}): {count_enfermo:,} images positive for {disease_name.lower()}.")
    print(f"    \\end{{itemize}}")
    print("")