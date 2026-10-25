# Prácticas de Panificación y Pastelería (9057) · Universidad de Burgos

Herramientas para las prácticas de laboratorio del Grado en Ciencias Gastronómicas.

Prof. Kerly Palacios · Departamento de Biotecnología y Ciencias de los Alimentos

**Acceso:** https://kerlypt05.github.io/UBU_Practicas_Cccias/ · las páginas piden la clave del espacio de prácticas, que se publica en UBUVirtual.

| Paso | Recurso | Práctica | Carpeta |
|---|---|---|---|
| 1 | Análisis de la miga (aplicativo de Colab) | PR-3 · Bizcochos de espuma | `pr3-miga/` |
| 2 | Simulación teórica vs datos experimentales (simulador) | PR-3 · Bizcochos de espuma | `pr3-espumas/` |

El material de apoyo a las clases teóricas está en otro espacio, con su propia clave: https://kerlypt05.github.io/UBU_Docencia_Ccias/

## Organización

- Cada recurso tiene su carpeta con una página de acceso (`index.html`); el simulador, además, `simulador.html`. La guía del aplicativo de miga (PDF y Word) va dentro de su página de acceso.
- Las páginas están **cifradas** (AES-256-GCM con clave derivada por PBKDF2). Las fuentes en claro se conservan fuera del repositorio.
- El código del aplicativo de miga (`pr3-miga/aplicativo_analisis_alveolar_PR3.py`), su cuaderno de Colab y `version.json` quedan **abiertos**: el cuaderno los descarga desde aquí al ejecutarse.
- `assets/logo_ubu.png`: logo de la Universidad de Burgos.

## Mantenimiento

- **Cifrar o cambiar la clave:** `python3 herramientas/cifrar_paginas.py --fuentes CARPETA_FUENTES/practicas --salida . --clave "CLAVE" --tema azul`
- **Actualizar la guía del aplicativo de miga:** sustituir `practicas/pr3-miga/descargas/*.docx` y `*.pdf` en la carpeta de fuentes (mismo nombre), ejecutar `python3 _generar_paginas.py` y volver a cifrar.
- **Nueva versión del aplicativo de miga:** ver `pr3-miga/CHANGELOG.md` (el código no se cifra).

Requisito: Python 3 con `pip install cryptography`.
