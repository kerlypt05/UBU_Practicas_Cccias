# Historial de versiones · Aplicativo de análisis de miga PR-3

La versión vigente es la que figura en `version.json`. El cuaderno de Colab descarga siempre esa versión.

## 1.0 · 23/10/2026

- 25/10/2026: el aplicativo pasa al espacio de prácticas (repositorio UBU_Practicas_Cccias). El código no cambia.
- Primera versión publicada para los estudiantes.
- Blanco control y validación por réplicas; grupos y réplicas con nombre libre, sin límite de grupos.
- Resolución de análisis común de 20 px/mm; detección de la regla y de la ventana de 3 × 3 cm.
- Comparación entre grupos: recorridos sin solape (criterio del guion) y Tukey HSD como criterio complementario.
- Nueva variable: % del área en alvéolos > 2 mm.
- Reporte Hoja D · Miga en Word (formato UBU); la descarga ya no incluye archivos CSV.

## Cómo publicar una versión nueva

1. Sustituir `aplicativo_analisis_alveolar_PR3.py` por el código nuevo.
2. Actualizar `version.json` (número, fecha y cambios) y añadir la entrada en este historial.
3. Hacer el commit en `main` y crear la etiqueta `pr3-miga-vX.Y`.

El cuaderno, la página y las copias que ya tengan los estudiantes pasan a usar la versión nueva sin más cambios.
