# Historial de versiones · Aplicativo de análisis de miga PR-3

La versión vigente es la que figura en `version.json`. El cuaderno de Colab descarga siempre esa versión.

## 1.1 · 06/11/2026

**Comparación entre grupos.** Se retira la discriminación por significación estadística. Con tres réplicas de una sola pieza por grupo no cabe una prueba de hipótesis que merezca ese nombre, y declarar «significativo / no significativo» induce a leer la práctica como un experimento de inferencia que no es. En su lugar:

- Desaparecen el ANOVA, el Tukey HSD, las letras de agrupación y los veredictos «diferencia sólida / indicio / no se puede afirmar».
- El resultado se presenta como **media ± DE** de cada grupo, con las tres réplicas a la vista.
- Se añade la **LSD de Fisher como regla de lectura**, dibujada como barra-patrón junto a las medias y dada par a par en la tabla, junto con la razón continua `diferencia / LSD`. Sin corrección por comparaciones múltiples: es deliberadamente permisiva.
- La LSD se calcula **par a par** (varianza común de los dos grupos comparados, gl = nA + nB − 2), no agrupando todos los grupos. Agrupar haría que una sola pieza irregular ensanchase la regla para todas las comparaciones, incluidas las de dos piezas uniformes.
- El solapamiento de recorridos se conserva como dato descriptivo, porque es la regla que usa la Hoja E del libro de registro, pero ya no emite veredicto.

**Densidad alveolar como variable primaria.** Encabeza figura, matriz y reporte; el área alveolar pasa a variable de apoyo y va siempre al lado, porque muchos alvéolos pequeños y pocos grandes pueden dar la misma área.

**Heterogeneidad por técnica.** Nueva lectura y nueva figura: CV entre las réplicas de cada pieza, con su límite de aceptación. Separa lo que aporta la fórmula de lo que aporta la elaboración.

**Figura de comparación.** Deja de usar barras (el eje no arranca en cero y una barra truncada exagera las diferencias): rombo de media, bigote de ± DE y réplicas a la vista. Decimales adaptados a la magnitud de cada variable.

**Reporte de resultados · Miga** (antes «Reporte Hoja D · Miga»). Se simplifica: una sola tabla con el blanco y las réplicas de cada grupo en el orden de la Hoja D (escala, área alveolar, densidad, Feret P90, % < 0,5 mm, % 0,5–1 mm), una línea de validación y las tres figuras de la comparación —medias ± DE, heterogeneidad y matriz de pares—. Las medias, los recorridos y los CV se dejan fuera: los calcula el libro de registro. Se genera en PDF y en Word desde un único generador de contenido.

**Protocolo de uso.** El cuaderno de Colab pasa a un diagrama de cinco pasos, sin explicación extendida: el procedimiento vive en la guía de uso, que se descarga de la página de acceso. La guía (v4) pierde los cuadros de registro y la pregunta de interpretación, que están en el libro de registro.

**Página de acceso.** Las novedades describen la lectura estadística nueva; se retira el bloque con el enlace al repositorio y al historial de versiones; se sustituye la descarga de la guía en Word por la del libro de registro en Excel.

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
3. Si cambia el protocolo, sustituir también `Aplicativo_Analisis_alveolar_PR3.ipynb`, actualizar la guía de `descargas/` y volver a cifrar `index.html` desde la fuente en claro.
4. Hacer el commit en `main` y crear la etiqueta `pr3-miga-vX.Y`.

El cuaderno, la página y las copias que ya tengan los estudiantes pasan a usar la versión nueva sin más cambios.
