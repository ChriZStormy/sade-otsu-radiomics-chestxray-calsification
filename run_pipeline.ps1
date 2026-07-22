Write-Host "Iniciando Pipeline Completo (ISCI 2026)..."

Write-Host "1. Ejecutando extracción de características (Brain)..."
python datasets_brain.py

Write-Host "2. Ejecutando extracción de características (Tórax)..."
python datasets_torax.py

Write-Host "3. Renombrando archivos para análisis estadístico..."
python renamer.py

Write-Host "4. Ejecutando análisis estadístico maestro..."
python clasificacion.py

Write-Host "¡Pipeline finalizado exitosamente!"
