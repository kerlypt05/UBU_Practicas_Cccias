# Aplicativo de análisis alveolar de miga · PR-3 · Panificación y Pastelería (9057) · Universidad de Burgos
# Versión 1.0 · 23/10/2026 · Código abierto: https://github.com/kerlypt05/UBU_Practicas_Cccias/tree/main/pr3-miga
# Este archivo lo descarga y ejecuta el cuaderno de Colab de la misma carpeta.

# ============================================================
#  APLICATIVO DE ANÁLISIS ALVEOLAR DE MIGA
#  PR-3 · Bizcochos de espuma · Panificación y Pastelería (UBU 9057)
#  Prof. Kerly Palacios · Universidad de Burgos
#
#  Código abierto: puede leerse, copiarse y modificarse libremente.
#  Deriva del aplicativo de PR-1 / PR-2 (miga de pan). Cambios de PR-3:
#    · parámetros del método reescalados a la miga de bizcocho (manual P4 §7.1)
#    · umbral con contraste mínimo, para que un blanco devuelva ~0 % de área
#    · validación por blanco control (cartulina gris)
#    · validación por réplicas (CV y controles de plausibilidad)
#    · comparación entre piezas por no solapamiento de recorridos
#  Referencias de diseño: 9057_PR3_P4_manual_profesor_v1 §7,
#  9057_PR3_P0_arquitectura_v2 §4, 9057_PR3_revision_tecnica_v1 §4.
# ============================================================
import os, glob, json, zipfile, warnings, itertools
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
from skimage.segmentation import watershed
warnings.filterwarnings("ignore")

PRACTICA = "PR-3"
PRODUCTO = "bizcochos de espuma"

# ------------------------------------------------------------
#  PARÁMETROS DEL MÉTODO (en mm: no dependen de la resolución de la foto)
#  Valor PR-2 (pan) → valor PR-3 (bizcocho). El alveolado de un bizcocho de
#  espuma es unas 4-5 veces más fino que el de un pan.
#  HIPÓTESIS: reescalados por tamaño característico; se confirman con el piloto.
# ------------------------------------------------------------
VENTANA_MM = 30.0     # ventana de la plantilla 3 × 3 cm: tamaño de referencia que valida la escala
LADO_ROI_MM = 24.0    # zona analizada 24 × 24 mm (sin cambio: más alvéolos, menos error de muestreo)
FONDO_MM = 2.0        # 5,0  → 2,0   sigma de la sustracción de fondo (varias veces la celda)
APERTURA_MM = 0.03    # 0,065 → 0,03 radio de la apertura morfológica
HUECO_MM2 = 0.02      # 0,07 → 0,02  área máxima de hueco que se rellena dentro de un alvéolo
WS_H_MM = 0.04        # 0,10 → 0,04  altura de los máximos del watershed
WS_SUAV_MM = 0.03     # 0,065 → 0,03 suavizado previo al watershed
DMIN_MM = 0.2         # 0,5  → 0,2   diámetro equivalente mínimo contado (opción recomendada)
LADO_MAX_PX = 3000    # 2048 → 3000  las fotos mayores se reducen; hacen falta ≥ 25 px/mm
CONTRASTE_MIN = 8.0   # NUEVO. Una zona oscura solo es alvéolo si, en promedio, es al menos 8 niveles
                      # de gris (de 255) más oscura que su entorno. Sin este mínimo, una superficie
                      # lisa (el blanco) se partiría siempre en dos mitades: Otsu siempre encuentra un umbral.

# Clases por diámetro de Feret máximo (mm). PR-2: < 2 / 2-5 / > 5 mm. AJUSTE CRÍTICO.
CLASES = [(0.0, 0.5, "Finos < 0,5 mm"), (0.5, 1.0, "Medios 0,5-1 mm"), (1.0, np.inf, "Grandes > 1 mm")]
LIM_MUY_GRANDES_MM = 2.0      # % del ÁREA alveolar en alvéolos con Feret > 2 mm: recoge los huecos grandes
                              # (coalescencia) que dentro de «Grandes > 1 mm» quedan diluidos. Se lee por área,
                              # no por número: por número son solo un 3-7 % de los alvéolos (10-25 por foto).
COLOR_CLASE_BGR = {0: (255, 170, 0), 1: (0, 140, 255), 2: (0, 0, 230)}
COLOR_CLASE = ["#00aaff", "#ff8c00", "#e60000"]   # mismos colores que los contornos
HIST_PASO_MM = 0.1    # anchura de los intervalos del histograma de tamaños
DETALLE_MM = 6.0      # lado del recorte ampliado de la figura de segmentación

# ------------------------------------------------------------
#  CRITERIOS DE ACEPTACIÓN (revisión técnica §4)
#  HIPÓTESIS: umbrales iniciales, a fijar con el piloto.
# ------------------------------------------------------------
BLANCO_AREA_OK = 2.0          # % de área alveolar en el blanco: ≤ 2 % válido
BLANCO_AREA_MAX = 5.0         # 2-5 % revisar; > 5 % no válido
CV_MAX = {"area_alveolar_%": 10.0, "densidad_alv_cm2": 20.0, "escala_foto_px_mm": 5.0}
AREA_PLAUSIBLE = (15.0, 75.0) # % de área alveolar fuera de este intervalo: sospecha de artefacto
REL_EJES_MAX = 2.5            # contornos alargados: grietas del corte o watershed mal separado
FERET_P90_MAX_FRAC = 0.25     # Feret P90 > 25 % del lado de la zona: fusión masiva de alvéolos
ROI_TOL = 0.10                # área analizada ±10 % de la nominal (24 × 24 mm = 5,76 cm²)
RES_ANALISIS = 20.0           # px/mm: TODAS las fotos se analizan a esta resolución (las de más resolución
                              # se reducen). Sin esto, el recuento y los % por clase dependen de la distancia
                              # de la cámara: con fotos reales de PR-3 la densidad seguía a la escala (r = 0,89).
                              # Con fotos originales de cámara (≥ 40 px/mm) puede subirse a 25-30; siempre igual
                              # para todos los grupos que se comparen.
RES_OK, RES_MIN = RES_ANALISIS, 15.0   # px/mm de la foto original: ≥ 20 adecuada; 15-20 revisar; < 15 no válida
RES_TOL = 0.05                # una foto hasta un 5 % por debajo de RES_ANALISIS se analiza tal cual
TOL_VENTANA = 0.15            # regla y ventana de 3 cm (tamaño de referencia) deben coincidir ±15 %
REGLA_REG_MIN = 0.5           # regularidad mínima de la regla (autocorrelación a 1 mm)
REGLA_ERR_MAX = 0.16          # error medio de posición de las marcas, en fracción de 1 mm
BLANCO_DE_MIN = 0.5           # DE de la cartulina por debajo: no es una foto (relleno digital)
CONTRASTE_BAJO = 10.0         # DE del contraste local (niveles de gris) por debajo: luz poco rasante


def _nombres_clase():
    return [nom for _, _, nom in CLASES]


def _corto(nom):
    """'Finos < 0,5 mm' -> 'finos'."""
    return nom.split()[0].lower()


def _rellenar_huecos(m, area):
    """Compatible con versiones antiguas y nuevas de scikit-image."""
    try:
        return morphology.remove_small_holes(m, max_size=area)
    except TypeError:
        return morphology.remove_small_holes(m, area_threshold=area)


def _factor(img):
    """Tamaño de la foto respecto a 2048 px: escala los parámetros que van en píxeles
    (detección de ventana y de regla) para que se comporten igual que en PR-2."""
    return max(img.shape[:2]) / 2048.0


def _nucleo(n, s):
    n = max(3, int(round(n * s)) | 1)
    return np.ones((n, n), np.uint8)


# ---------- 1. Ventana de miga ----------
def _mascaras_miga(img):
    """Máscaras candidatas de la miga dentro de la ventana: por tono (miga frente a plantilla roja),
    miga blanca, miga poco saturada y solo brillo. Se queda la que da la mejor ventana."""
    s = _factor(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    mascaras = []
    if S.mean() > 15:   # foto en color
        Hh = hsv[..., 0]   # tono en OpenCV: 0-180; rojo ≈ 0-7 y 170-180, amarillo-crema-tostado ≈ 8-45
        # Por tono: la miga amarilla de una genovesa está tan saturada como la plantilla roja, pero
        # su tono es otro. Sirve para miga blanca, crema, amarilla o tostada frente a plantilla roja u oscura.
        mascaras.append((((Hh >= 8) & (Hh <= 45)) | (S < 60)) & (V > 100))
        mascaras += [(S < 70) & (V > 120), (S < 140) & (V > 110)]
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 3 * s)
    mascaras.append(g > filters.threshold_otsu(g))
    return [m.astype(np.uint8) for m in mascaras]


def _mejor_componente(mask, min_frac=0.05):
    """Componente clara, grande, casi cuadrada, compacta y centrada. Puede tocar el borde de la
    foto (foto recortada a la ventana), pero se penaliza: así la mesa o el fondo claro no ganan.
    Devuelve (componente, puntuación)."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    H, W = mask.shape
    mejor, pts = None, -1
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < min_frac * H * W:
            continue
        cuad = min(w, h) / max(w, h)
        if cuad < 0.7:
            continue
        solidez = a / float(w * h)
        dist = np.hypot(cen[k][0] - W / 2, cen[k][1] - H / 2) / np.hypot(W, H)
        bordes = int(x <= 1) + int(y <= 1) + int(x + w >= W - 1) + int(y + h >= H - 1)
        p = a * cuad * min(1.0, solidez) * (1 - dist) * 0.5 ** bordes
        if p > pts:
            mejor, pts = k, p
    if mejor is None:
        return None, -1
    return (lab == mejor).astype(np.uint8), pts


def _componente_ventana(img):
    s = _factor(img)
    k1, k2 = _nucleo(25, s), _nucleo(41, s)
    mejor = (None, -1, None)
    for m in _mascaras_miga(img):
        limpia = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k1), cv2.MORPH_OPEN, k2)
        comp, p = _mejor_componente(limpia)
        if comp is not None and p > mejor[1]:
            mejor = (comp, p, m)
    return mejor[0], mejor[2]


def _medir_ventana(mask, cx, cy, half):
    """Desde un cuadrado inicial centrado en (cx, cy), extiende cada lado mientras la franja
    que se añade siga siendo miga (≥ 50 %). Se detiene en el borde de la plantilla o en el de la
    foto (lado truncado). Devuelve (x0, x1, y0, y1, truncado_x, truncado_y)."""
    H, W = mask.shape
    x0, x1 = max(0, cx - half), min(W - 1, cx + half)
    y0, y1 = max(0, cy - half), min(H - 1, cy + half)
    m = mask.astype(np.float32)
    def crecer(pos, paso, limite, franja):
        while pos != limite and franja(pos + paso).mean() >= 0.5:
            pos += paso
        return pos
    x0 = crecer(x0, -1, 0, lambda c: m[y0:y1 + 1, c])
    x1 = crecer(x1, 1, W - 1, lambda c: m[y0:y1 + 1, c])
    y0 = crecer(y0, -1, 0, lambda r: m[r, x0:x1 + 1])
    y1 = crecer(y1, 1, H - 1, lambda r: m[r, x0:x1 + 1])
    return x0, x1, y0, y1, (x0 == 0 or x1 == W - 1), (y0 == 0 or y1 == H - 1)


def detectar_ventana(img, pxmm=None, ventana_mm=VENTANA_MM):
    """Localiza la ventana de la plantilla y mide su lado (el tamaño de referencia de 3 cm).
    Con escala conocida busca el cuadrado de 30 mm con más miga dentro de la componente
    elegida (así el brillo de la plantilla pegado a la ventana no la deforma).
    Devuelve dict(mascara, centro, lado_px, truncada) o None."""
    comp, raw = _componente_ventana(img)
    if comp is None:
        return None
    H, W = comp.shape
    if pxmm:
        L = int(round(min(ventana_mm * pxmm, H, W)))
        caja = cv2.boxFilter((raw & comp).astype(np.float32), -1, (L, L), normalize=True,
                             borderType=cv2.BORDER_CONSTANT)
        caja[comp == 0] = 0
        cy, cx = np.unravel_index(int(np.argmax(caja)), caja.shape)
        half = max(2, int(0.35 * L))
    else:
        d = cv2.distanceTransform(comp, cv2.DIST_C, 3)
        cy, cx = np.unravel_index(int(np.argmax(d)), d.shape)
        half = max(2, int(0.7 * d.max()))
    x0, x1, y0, y1, tx, ty = _medir_ventana(raw, int(cx), int(cy), half)
    mascara = np.zeros_like(comp); mascara[y0:y1 + 1, x0:x1 + 1] = 1
    lados = [v for v, t in ((x1 - x0 + 1, tx), (y1 - y0 + 1, ty)) if not t]
    return dict(mascara=mascara, centro=((x0 + x1) // 2, (y0 + y1) // 2),
                lado_px=float(np.mean(lados)) if lados else None, truncada=(tx, ty),
                rect=(x0, x1, y0, y1))


def centro_cuadrado_inscrito(ventana, pxmm):
    d = cv2.distanceTransform(ventana.astype(np.uint8), cv2.DIST_C, 3)
    ys, xs = np.where(d >= d.max() - 1)
    return (int(xs.mean()), int(ys.mean())), 2 * d.max() / pxmm


# ---------- 2. Escala con la regla: lectura de las marcas de milímetro ----------
def _perfil(franja_gris, eje, sigma=12):
    """Perfil 1D de una franja: paso alto a lo largo de la regla y media en el eje
    transversal. No depende del color ni de si las marcas son claras u oscuras."""
    g = franja_gris.astype(float)
    if eje == "x":
        g = g.T
    hp = g - cv2.GaussianBlur(g, (0, 0), sigmaX=0.1, sigmaY=sigma)
    p = hp.mean(1)
    return p - p.mean()


def _periodo_fft(p, pmin=4, pmax=200):
    n = len(p)
    if n < 12 * pmin:
        return None
    N = 1 << int(np.ceil(np.log2(n * 8)))
    F = np.abs(np.fft.rfft(p * np.hanning(n), N)); f = np.fft.rfftfreq(N)
    ok = (f > 1 / pmax) & (f < 1 / pmin)
    if not ok.any():
        return None
    return 1 / f[ok][int(np.argmax(F[ok]))]


def _marcas(p, p0):
    """Posiciones de las marcas (máximos del perfil, o mínimos si son oscuras) y ajuste lineal
    posición = a + periodo × índice. Tolera marcas perdidas, números y espaciado irregular de
    una foto comprimida. Devuelve (periodo, n_marcas, error relativo, posiciones) o None."""
    from scipy.signal import find_peaks
    mejor = None
    for signo in (1, -1):
        q = signo * p
        pk, _ = find_peaks(q, distance=max(2, int(0.6 * p0)), prominence=0.25 * np.std(q))
        if len(pk) < 8:
            continue
        idx = np.round((pk - pk[0]) / p0)
        for _ in range(2):                                 # ajuste y recalculo de índices
            b, a = np.polyfit(idx, pk, 1)
            idx = np.round((pk - a) / b)
        res = pk - (a + b * idx)
        ok = np.abs(res) < 0.3 * b                         # descarta números u otros picos sueltos
        if ok.sum() < 8:
            continue
        b, a = np.polyfit(idx[ok], pk[ok], 1)
        err = float(np.sqrt(np.mean((pk[ok] - (a + b * idx[ok])) ** 2)) / b)
        cobertura = ok.sum() / (idx[ok].max() - idx[ok].min() + 1)
        cand = (b, int(ok.sum()), err, cobertura, pk[ok])
        if mejor is None or cand[1] * cand[3] > mejor[1] * mejor[3]:
            mejor = cand
    return mejor


def franjas_candidatas(shape, ventana=None, s=1.0):
    """Franjas estrechas que barren toda la foto en vertical y en horizontal (la regla puede
    estar en cualquier lado, también pegada al borde de una foto recortada), más las franjas
    junto a los cuatro lados de la ventana si se conoce. Cada una: (nombre, x0, x1, y0, y1, eje)."""
    H, W = shape[:2]
    out = []
    w = max(14, int(0.035 * W))
    for a in range(0, W - w + 1, max(4, w // 3)):
        out.append((f"vertical x={a}", a, a + w, 0, H, "y"))
    w = max(14, int(0.035 * H))
    for a in range(0, H - w + 1, max(4, w // 3)):
        out.append((f"horizontal y={a}", 0, W, a, a + w, "x"))
    if ventana is not None:
        x0, x1, y0, y1 = ventana["rect"]; w = max(14, int(0.035 * max(H, W)))
        out += [("izquierda", max(0, x0 - 2 * w), max(1, x0 - 2), 0, H, "y"),
                ("derecha", min(W - 2, x1 + 2), min(W, x1 + 2 * w), 0, H, "y"),
                ("arriba", 0, W, max(0, y0 - 2 * w), max(1, y0 - 2), "x"),
                ("abajo", 0, W, min(H - 2, y1 + 2), min(H, y1 + 2 * w), "x")]
    return [c for c in out if c[2] - c[1] > 8 and c[4] - c[3] > 8]


def escala_regla(img, ventana=None):
    """Escala (px/mm) a partir de las marcas de milímetro de la regla, en escala de grises.
    Se acepta si hay al menos 12 marcas (más de 1 cm) con espaciado regular: regularidad
    ≥ REGLA_REG_MIN y error de posición ≤ REGLA_ERR_MAX del periodo (las fotos comprimidas
    desplazan cada marca ±1-2 px; el ajuste lineal sobre todas ellas lo compensa).
    Devuelve (px_mm, info) o (None, info)."""
    s = _factor(img)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma = max(6.0, 12 * s)
    mejor = None
    for nom, x0, x1, y0, y1, eje in franjas_candidatas(img.shape, ventana, s):
        p = _perfil(gris[y0:y1, x0:x1], eje, sigma)
        p0 = _periodo_fft(p)
        if not p0:
            continue
        m = _marcas(p, p0)
        if m is None:
            continue
        per, n, err, cob, pos = m
        # Regularidad: autocorrelación del perfil a una distancia de 1 mm. Una regla da 0,6-0,95;
        # la textura de la miga, la mesa o el ruido dan ≈ 0 aunque produzcan picos sueltos.
        ac = np.correlate(p, p, "full")[len(p) - 1:]; ac = ac / (ac[0] + 1e-12)
        j = int(round(per)); reg = float(ac[max(1, j - 1):j + 2].max()) if j + 1 < len(ac) else 0.0
        punt = n * cob * max(0.0, reg) * max(0.0, 1 - 3 * err)
        d = dict(franja=nom, caja=(x0, x1, y0, y1), eje=eje, px_mm=per, marcas=n, error=err,
                 cobertura=cob, regularidad=reg, calidad=punt, perfil=p, posiciones=pos)
        if mejor is None or punt > mejor["calidad"]:
            mejor = d
    if mejor is None:
        return None, None
    if (mejor["marcas"] < 12 or mejor["regularidad"] < REGLA_REG_MIN or mejor["error"] > REGLA_ERR_MAX
            or mejor["cobertura"] < 0.6):
        return None, mejor
    return mejor["px_mm"], mejor


# ---------- 3-5. ROI, 8 bits + fondo, umbral con contraste mínimo + watershed ----------
def preparar(img, pxmm, centro, lado_mm, ventana):
    """Recorta la zona de análisis y resta el fondo. Devuelve:
    color, máscara de la zona, imagen de 8 bits (para ver), contraste local en niveles
    de gris (para medir) y fracción de la zona que cae dentro de la ventana."""
    half = int(round(lado_mm / 2 * pxmm)); cx, cy = centro
    H, W = img.shape[:2]
    if cx - half < 0 or cy - half < 0 or cx + half > W or cy + half > H:
        raise ValueError("La zona de análisis se sale de la foto.")
    roi = np.zeros((H, W), np.uint8); roi[cy - half:cy + half, cx - half:cx + half] = 1
    dentro = 1.0
    if ventana is not None:
        dentro = (roi & ventana).sum() / roi.sum(); roi &= ventana
    x, y, w, h = cv2.boundingRect(roi)
    color = img[y:y + h, x:x + w]; roi = roi[y:y + h, x:x + w].astype(bool)
    gris = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(float)
    contraste = gris - cv2.GaussianBlur(gris, (0, 0), FONDO_MM * pxmm)   # niveles de gris respecto al entorno
    v = contraste[roi]
    plano = np.clip((contraste - v.min()) / (v.max() - v.min() + 1e-9) * 255, 0, 255)
    return color, roi, plano, contraste, dentro


def segmentar(contraste, roi, pxmm, metodo="otsu"):
    """Alvéolo = zona más oscura que el umbral automático cuya profundidad MEDIA es de al menos
    CONTRASTE_MIN niveles de gris por debajo de su entorno. El criterio se aplica a cada zona
    entera, no a cada píxel: no recorta el borde de los alvéolos reales y elimina las manchas de
    ruido de una superficie lisa. Devuelve (etiquetas, umbral automático, % de área descartada)."""
    t = {"otsu": filters.threshold_otsu, "li": filters.threshold_li,
         "media": filters.threshold_mean}[metodo](contraste[roi])
    alv = (contraste < t) & roi
    alv = ndi.binary_opening(alv, structure=morphology.disk(max(1, round(APERTURA_MM * pxmm))))
    alv = _rellenar_huecos(alv, int(HUECO_MM2 * pxmm ** 2))
    zonas, nz = ndi.label(alv)
    descartada = 0.0
    if nz:
        prof = np.asarray(ndi.mean(contraste, zonas, index=np.arange(1, nz + 1)))
        validas = np.r_[False, prof <= -CONTRASTE_MIN]
        nuevo = validas[zonas]
        descartada = 100 * (alv.sum() - nuevo.sum()) / max(roi.sum(), 1)
        alv = nuevo
    dist = ndi.distance_transform_edt(alv)
    marcas = measure.label(morphology.h_maxima(ndi.gaussian_filter(dist, max(1.0, WS_SUAV_MM * pxmm)),
                                               max(1.0, WS_H_MM * pxmm)))
    return watershed(-dist, marcas, mask=alv, watershed_line=True), float(t), float(descartada)


# ---------- 6-7. Medición y clasificación ----------
COLS_ALV = ["etiqueta", "area_mm2", "deq_mm", "feret_mm", "rel_ejes", "clase", "k", "borde_insesgado"]


def _clase(feret):
    for k, (a, b, _) in enumerate(CLASES):
        if a <= feret < b:
            return k
    return len(CLASES) - 1


def medir(lab, pxmm, dmin):
    H, W = lab.shape; filas = []
    for q in measure.regionprops(lab):
        area = q.area / pxmm ** 2; deq = 2 * np.sqrt(area / np.pi)
        if deq < dmin:
            continue
        r0, c0, r1, c1 = q.bbox; fe = q.feret_diameter_max / pxmm; k = _clase(fe)
        filas.append(dict(etiqueta=q.label, area_mm2=area, deq_mm=deq, feret_mm=fe,
                          rel_ejes=q.major_axis_length / max(q.minor_axis_length, 1),
                          clase=CLASES[k][2], k=k, borde_insesgado=bool(r1 >= H - 1 or c1 >= W - 1)))
    return pd.DataFrame(filas, columns=COLS_ALV)


def resumir(df, roi, pxmm, contraste=None):
    a = roi.sum() / pxmm ** 2; c = df[~df.borde_insesgado.astype(bool)]; n = len(c)
    r = dict(escala_px_mm=round(pxmm, 2), roi_cm2=round(a / 100, 2), N=n,
             densidad_alv_cm2=round(n / (a / 100), 1), N_equiv_3x3=round(9 * n / (a / 100)))
    for i, (_, _, nom) in enumerate(CLASES):
        ni = int((c.k == i).sum()); r[f"n {nom}"] = ni; r[f"% {nom}"] = round(100 * ni / n, 1) if n else np.nan
    at = df.area_mm2.sum()
    r["%_area_gt2mm"] = round(100 * df.area_mm2[df.feret_mm > LIM_MUY_GRANDES_MM].sum() / at, 1) if at > 0 else np.nan
    r.update({"area_alveolar_%": round(100 * df.area_mm2.sum() / a, 1),
              "area_media_mm2": round(c.area_mm2.mean(), 3) if n else np.nan,
              "feret_mediano_mm": round(c.feret_mm.median(), 2) if n else np.nan,
              "feret_p90_mm": round(c.feret_mm.quantile(0.9), 2) if n else np.nan,
              "rel_ejes_mediana": round(c.rel_ejes.median(), 2) if n else np.nan})
    if contraste is not None:
        r["contraste_DE"] = round(float(np.std(contraste[roi])), 1)
    return r


def superponer(color, lab, df, pxmm):
    ov = color.copy(); val = dict(zip(df.etiqueta, df.k))
    grosor = 1 if pxmm < 40 else 2
    for q in measure.regionprops(lab):
        if q.label in val:
            r0, c0, _, _ = q.bbox
            cs, _ = cv2.findContours(q.image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, [cc + np.array([[c0, r0]]) for cc in cs], -1, COLOR_CLASE_BGR[val[q.label]], grosor)
    return ov


def _recorte_central(im, pxmm, lado_mm):
    h, w = im.shape[:2]; m = int(round(lado_mm * pxmm / 2)); cy, cx = h // 2, w // 2
    return im[max(0, cy - m):cy + m, max(0, cx - m):cx + m]


def figura(paneles, titulo, lado_mm, ruta=None):
    """Por réplica: zona en color, 8 bits con fondo restado, contornos por clase y
    detalle ampliado del centro (para comprobar a simple vista que los contornos
    caen sobre alvéolos y no sobre pared)."""
    n = len(paneles)
    fig, ax = plt.subplots(n, 4, figsize=(16, 4.1 * n + 0.7), squeeze=False)
    for r, (etq, color, plano, ov, pxmm) in enumerate(paneles):
        det = _recorte_central(ov, pxmm, DETALLE_MM)
        ims = [cv2.cvtColor(color, cv2.COLOR_BGR2RGB), plano, cv2.cvtColor(ov, cv2.COLOR_BGR2RGB),
               cv2.cvtColor(det, cv2.COLOR_BGR2RGB)]
        for c in range(4):
            a = ax[r, c]; a.imshow(ims[c], cmap="gray" if c == 1 else None, interpolation="nearest" if c == 3 else None)
            a.set_xticks([]); a.set_yticks([])
            hh = ims[c].shape[0]; barra = 1 if c == 3 else 5
            a.plot([hh * 0.03, hh * 0.03 + barra * pxmm], [hh * 0.95] * 2, color="yellow", lw=4)
            a.text(hh * 0.03, hh * 0.91, f"{barra} mm", color="yellow", fontsize=9, weight="bold")
        if r == 0:
            # marca en el panel de contornos la zona que se amplía
            h3, w3 = ov.shape[:2]; m = DETALLE_MM * pxmm / 2
            ax[0, 2].add_patch(Rectangle((w3 / 2 - m, h3 / 2 - m), 2 * m, 2 * m, fill=False, ec="white", lw=1.5, ls="--"))
        ax[r, 0].set_ylabel(etq, fontsize=12)
    for a, t in zip(ax[0], [f"Zona central {lado_mm:g}×{lado_mm:g} mm (color)", "8 bits + sustracción de fondo",
                            "Umbral + watershed, clase por Feret", f"Detalle ampliado {DETALLE_MM:g}×{DETALLE_MM:g} mm"]):
        a.set_title(t, fontsize=11)
    fig.legend(handles=[Patch(color=COLOR_CLASE[k], label=nom) for k, nom in enumerate(_nombres_clase())],
               loc="lower center", ncol=3, fontsize=11, frameon=False)
    fig.suptitle(titulo, fontsize=14, weight="bold")
    H = 4.1 * n + 0.7
    plt.tight_layout(rect=[0, 0.6 / H, 1, 1 - 0.45 / H])
    if ruta:
        plt.savefig(ruta, dpi=150)
    return fig


# ---------- Mensajes con fondo propio (legibles en tema claro y oscuro) ----------
def _nota(texto, tipo="info"):
    col = {"info": ("#eef4fc", "#1c5cab"), "ok": ("#e9f6ee", "#1b6b3a"),
           "aviso": ("#fff6e5", "#8a5300"), "error": ("#fdecec", "#a1251f")}[tipo]
    return (f"<div style='background:{col[0]};color:{col[1]};padding:8px 12px;border-radius:6px;"
            f"font-size:13px;margin:4px 0;line-height:1.45'>{texto}</div>")


def _html(texto, tipo="info"):
    from IPython.display import display, HTML
    display(HTML(_nota(texto, tipo)))


def _archivo_seguro(t):
    import re
    return re.sub(r"[^\w\-]+", "_", t).strip("_") or "muestra"


# ---------- Lectura de fotos ----------
EXT_IMG = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def leer_foto(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo abrir «{os.path.basename(ruta)}». Use JPG o PNG "
                         "(las fotos HEIC del iPhone deben exportarse antes a JPG).")
    f = LADO_MAX_PX / max(img.shape[:2])
    if f < 1:
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return img


# ---------- Opciones en lenguaje natural ----------
def _numero(texto):
    """Extrae el primer número de un texto ('31,6 px/mm' -> 31.6). None si no hay."""
    import re
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(texto))
    return float(m.group(0).replace(",", ".")) if m else None


def interpretar_opciones(zona, umbral, minimo, escala):
    lado = _numero(zona) or LADO_ROI_MM
    u = umbral.lower()
    met = "li" if u.startswith("li") else ("media" if u.startswith("media") else "otsu")
    dmin = _numero(minimo) or DMIN_MM
    esc = None
    if escala and escala.strip() and not escala.lower().startswith("auto"):
        esc = _numero(escala)
        if esc and "px/cm" in escala.lower().replace(" ", ""):
            esc = esc / 10.0
    return dict(lado=lado, umbral=met, dmin=dmin, escala=esc)


def configuracion(opciones):
    """Texto que identifica las condiciones de análisis: solo son comparables piezas con la misma."""
    return (f"zona {opciones['lado']:g} mm · umbral {opciones['umbral']} · mínimo {opciones['dmin']:g} mm · "
            f"fondo {FONDO_MM:g} · apertura {APERTURA_MM:g} · contraste mín. {CONTRASTE_MIN:g}")


def figura_control(img, ventana, info, centro, pxmm, lado, nombre, aviso, ruta=None):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw=dict(width_ratios=[1, 1.5]))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].set_title(nombre, fontsize=10)
    if ventana is not None:
        x0, x1, y0, y1 = ventana["rect"]
        ax[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="cyan", lw=1.5))
    if info:
        x0, x1, y0, y1 = info["caja"]
        ax[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=2))
    if centro and pxmm:
        h = lado / 2 * pxmm
        ax[0].add_patch(Rectangle((centro[0] - h, centro[1] - h), 2 * h, 2 * h, fill=False, ec="lime", lw=2))
    if info and info.get("perfil") is not None:
        p = info["perfil"]; ax[1].plot(p, color="0.3", lw=0.8)
        pos = info.get("posiciones")
        if pos is not None and len(pos):
            ax[1].plot(pos, p[pos], "o", color="red", ms=3.5, label=f"{len(pos)} marcas de 1 mm leídas")
            ax[1].legend(loc="upper right", fontsize=9, frameon=False)
        ax[1].set_title(f"Perfil de la regla: cada punto rojo debe caer sobre una marca de milímetro "
                        f"({info['px_mm']:.2f} px/mm)", fontsize=10)
        ax[1].set_xlabel("px a lo largo de la regla")
    else:
        ax[1].axis("off")
    import textwrap
    titulo = ("\n".join(textwrap.fill("⚠ " + t.strip(), 150) for t in aviso.split("⚠") if t.strip())
              if "⚠" in aviso else aviso)
    fig.suptitle(titulo + "\nCian = ventana de 3 cm · Amarillo = franja de la regla · Verde = zona analizada",
                 fontsize=11, color="red" if "⚠" in aviso else "black")
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta, dpi=90)
    return fig


# ---------- Análisis de una foto ----------


def analizar_foto(ruta, etq, opciones, base, mostrar_control="Solo si hay avisos"):
    """Escala, ventana, zona, segmentación y medida de una foto.
    Devuelve (resultado o None, tabla de alvéolos, panel de figura, texto de aviso)."""
    lado = opciones["lado"]
    nombre = os.path.basename(ruta); img = leer_foto(ruta)
    info, aviso, aviso_escala = None, "", False
    # 1 · Escala: regla (o la escrita por el analista)
    if opciones["escala"]:
        pxmm = opciones["escala"] * (min(1, LADO_MAX_PX / max(cv2.imread(ruta).shape[:2])))
        origen = "indicada"
    else:
        pxmm, info = escala_regla(img); origen = "regla"
    # 2 · Ventana de 3 cm, buscada con la escala si ya se conoce
    ventana = detectar_ventana(img, pxmm)
    lado_v = ventana["lado_px"] if ventana else None
    # 3 · Respaldo: sin regla legible, la ventana de 3 cm da la escala (tamaño de referencia)
    if pxmm is None and lado_v:
        pxmm = lado_v / VENTANA_MM; origen = "ventana"
        aviso_escala = True
        aviso += (f"⚠ {etq}: no se pudo leer la regla; la escala se ha tomado del tamaño de la ventana de "
                  f"{VENTANA_MM / 10:g} cm ({pxmm:.1f} px/mm). Es una estimación: compruebe la regla en la figura de control. ")
    if pxmm is None:
        motivo = ("la ventana está cortada por el borde de la foto" if ventana else "tampoco se detectó la ventana")
        return None, None, None, (f"⚠ {etq} ({nombre}): no se pudo leer la regla y {motivo}, así que no hay "
                                  "escala. Use la foto original (sin recortar), con la regla junto a la ventana, "
                                  "o escriba la escala en «Opciones avanzadas».")
    # 4 · Control cruzado: regla frente a ventana de 3 cm
    if origen != "ventana" and lado_v:
        pxmm_v = lado_v / VENTANA_MM
        if abs(pxmm / pxmm_v - 1) > TOL_VENTANA:
            aviso_escala = True
            aviso += (f"⚠ {etq}: la escala de la regla ({pxmm:.1f} px/mm) no cuadra con la ventana de "
                      f"{VENTANA_MM / 10:g} cm ({pxmm_v:.1f} px/mm). Revise la figura de control o indique la escala. ")
    if ventana is None:
        h, w = img.shape[:2]; centro = (w // 2, h // 2)
        aviso += f"⚠ {etq}: no se detectó la ventana de la plantilla; se analiza el centro de la foto. "
    else:
        centro = ventana["centro"]
    if pxmm < RES_OK * (1 - RES_TOL):
        aviso += (f"⚠ {etq}: resolución de {pxmm:.1f} px/mm, por debajo de la de análisis ({RES_ANALISIS:g} px/mm): "
                  f"esta foto ve menos alvéolos pequeños que las demás. Use la foto original de la cámara: sin "
                  "recortar, sin captura de pantalla y sin enviarla por mensajería. ")
    if aviso:
        txt = aviso
    elif info:
        txt = f"✔ {etq}: escala {pxmm:.2f} px/mm (regla, {info['marcas']} marcas de 1 mm leídas)"
    else:
        txt = f"✔ {etq}: escala {pxmm:.2f} px/mm ({origen})"
    fc = figura_control(img, ventana, info, centro, pxmm, lado, nombre, txt,
                        base + f"_control_{_archivo_seguro(etq)}.png")
    if mostrar_control.startswith("Siempre") or (aviso and not mostrar_control.startswith("Nunca")):
        plt.show()
    else:
        plt.close(fc)
    mascara = ventana["mascara"] if ventana else None
    # 5 · Normalización de resolución: todas las fotos se analizan a RES_ANALISIS px/mm
    escala_foto = pxmm
    if pxmm > RES_ANALISIS:
        k = RES_ANALISIS / pxmm
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        if mascara is not None:
            mascara = cv2.resize(mascara, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        centro = (int(round(centro[0] * k)), int(round(centro[1] * k)))
        pxmm = RES_ANALISIS
    try:
        color, roi, plano, contraste, dentro = preparar(img, pxmm, centro, lado, mascara)
    except ValueError:
        return None, None, None, (f"⚠ {etq}: la zona de {lado:g} mm no cabe en la foto con esta escala "
                                  f"({pxmm:.1f} px/mm). Use la foto completa o pruebe la zona de 20 mm.")
    lab, t, desc = segmentar(contraste, roi, pxmm, opciones["umbral"])
    df = medir(lab, pxmm, opciones["dmin"])
    r = {"emplazamiento": etq, "archivo": nombre, "origen_escala": origen, "dentro_pct": 100 * dentro,
         "escala_foto_px_mm": round(escala_foto, 2),
         "umbral_auto": round(t, 1), "descartado_contraste_%": round(desc, 1),
         "aviso_escala": aviso_escala,
         "ventana_mm": round(lado_v / escala_foto, 1) if lado_v else np.nan, **resumir(df, roi, pxmm, contraste)}
    df.insert(0, "emplazamiento", etq)
    return r, df, (etq, color, plano, superponer(color, lab, df, pxmm), pxmm), txt



# ============================================================
#  TABLAS, VALIDACIÓN Y REGISTRO DE LA SESIÓN
# ============================================================
def _columnas():
    """Nombres de columna para el estudiante. Las primeras siguen el orden de la TABLA E del guion."""
    nc = _nombres_clase()
    cols = {"emplazamiento": "Réplica", "archivo": "Archivo", "escala_foto_px_mm": "Escala de la foto (px/mm)",
            "escala_px_mm": "Escala de análisis (px/mm)",
            "area_alveolar_%": "Área alveolar (%)", "densidad_alv_cm2": "Densidad (alv/cm²)",
            "feret_mediano_mm": "Feret mediano (mm)", "feret_p90_mm": "Feret P90 (mm)"}
    for nom in nc:
        cols[f"% {nom}"] = f"% {nom[0].lower() + nom[1:]}"
    cols["%_area_gt2mm"] = "% del área en alvéolos > 2 mm"
    cols.update({"ventana_mm": "Ventana medida (mm)", "roi_cm2": "Área analizada (cm²)", "N": "N alvéolos", "N_equiv_3x3": "N equiv. 3×3 cm"})
    for nom in nc:
        cols[f"n {nom}"] = f"n {_corto(nom)}"
    cols.update({"area_media_mm2": "Área media (mm²)", "rel_ejes_mediana": "Relación de ejes",
                 "contraste_DE": "Contraste local (DE)"})
    return cols


COLUMNAS = _columnas()
FILAS_RESUMEN = ("Media", "DE", "CV (%)", "Mín", "Máx")


def tabla_calculos(res):
    """Réplicas + media, DE, CV y recorrido (mín-máx) de cada variable numérica."""
    num = [c for c in res.columns if c in COLUMNAS and pd.api.types.is_numeric_dtype(res[c])]
    tabla = res.copy()
    if len(res) > 1:
        cv = {c: (100 * res[c].std() / res[c].mean() if res[c].mean() else np.nan) for c in num}
        tabla = pd.concat([tabla, pd.DataFrame([
            {"emplazamiento": "Media", **{c: res[c].mean() for c in num}},
            {"emplazamiento": "DE", **{c: res[c].std() for c in num}},
            {"emplazamiento": "CV (%)", **cv},
            {"emplazamiento": "Mín", **{c: res[c].min() for c in num}},
            {"emplazamiento": "Máx", **{c: res[c].max() for c in num}}])], ignore_index=True)
    tabla = tabla[[c for c in COLUMNAS if c in tabla.columns]].rename(columns=COLUMNAS)
    return tabla.round(2)


# ---------- Estados de validación ----------
ORDEN_ESTADO = {"OK": 0, "REVISAR": 1, "PENDIENTE": 2, "NO VÁLIDO": 3}
ESTILO_ESTADO = {"OK": ("#e9f6ee", "#1b6b3a", "✔"), "REVISAR": ("#fff6e5", "#8a5300", "!"),
                 "PENDIENTE": ("#eef4fc", "#1c5cab", "…"), "NO VÁLIDO": ("#fdecec", "#a1251f", "✖")}


def _peor(estados):
    estados = [e for e in estados if e]
    return max(estados, key=lambda e: ORDEN_ESTADO[e]) if estados else "OK"


def _es(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{d}f}".replace(".", ",")


def _cv(serie):
    s = pd.Series(serie).dropna()
    return 100 * s.std(ddof=1) / s.mean() if len(s) > 1 and s.mean() else np.nan


def _estado_cv(cv, lim):
    if np.isnan(cv):
        return "REVISAR"
    return "OK" if cv <= lim else ("REVISAR" if cv <= 2 * lim else "NO VÁLIDO")


def estado_blanco(area, de=None, pxmm=None):
    """Estado del blanco y motivo. Tres condiciones: área alveolar próxima a cero, que la imagen
    sea una fotografía real (la cartulina tiene grano y ruido de cámara) y resolución suficiente."""
    if de is not None and de < BLANCO_DE_MIN:
        return "NO VÁLIDO", ("La zona analizada tiene un valor de gris constante: no es una fotografía sino un "
                             "relleno digital. Un blanco editado no comprueba nada (ni la luz ni la cámara). "
                             "Fotografíe la cartulina gris real dentro de la plantilla.")
    if area > BLANCO_AREA_MAX:
        return "NO VÁLIDO", ("El método marca textura como alvéolo: ningún dato de miga es interpretable hasta "
                             "corregir la luz o la foto.")
    if area > BLANCO_AREA_OK:
        return "REVISAR", "El método marca algo de textura como alvéolo. Repita la foto del blanco con la misma luz."
    if pxmm is not None and pxmm < RES_OK:
        return "REVISAR", (f"Resolución de {_es(pxmm)} px/mm, por debajo de {RES_OK:g}: el blanco debe hacerse con "
                           "la misma cámara, distancia y archivo original que las piezas.")
    return "OK", "—"


def fila_blanco(blanco):
    """Fila de la tabla de validación que resume el blanco control de la sesión."""
    if not blanco:
        return dict(Control="Blanco control", Valor="no analizado en esta sesión",
                    Criterio=f"área alveolar ≤ {_es(BLANCO_AREA_OK)} %", Estado="PENDIENTE",
                    Acción="Fotografíe la cartulina gris con la plantilla y la regla, y analícela como «Blanco control».")
    return dict(Control="Blanco control",
                Valor=(f"{_es(blanco['area_alveolar_%'])} % de área · grano {_es(blanco.get('contraste_DE'))} · "
                       f"{blanco['fecha']}"),
                Criterio=(f"área ≤ {_es(BLANCO_AREA_OK)} % válido, ≤ {_es(BLANCO_AREA_MAX)} % revisar · "
                          f"foto real (grano ≥ {_es(BLANCO_DE_MIN)})"),
                Estado=blanco["estado"], Acción=blanco.get("motivo", "—"))


def validar_muestra(res, opciones, n_fotos):
    """Controles de la pieza, sin el blanco (revisión técnica §4.2-4.3)."""
    filas = []

    def add(control, valor, criterio, estado, accion="—"):
        filas.append(dict(Control=control, Valor=valor, Criterio=criterio, Estado=estado,
                          Acción=accion if estado != "OK" else "—"))

    n = len(res); lado = opciones["lado"]
    add("Réplicas analizadas", f"{n} de {n_fotos}", "3 réplicas por pieza",
        "OK" if n >= 3 else ("REVISAR" if n == 2 else "NO VÁLIDO"),
        "Con menos de 3 réplicas el recorrido y el CV pierden casi todo su valor. Suba las tres fotos de la pieza.")
    if "ventana_mm" in res and res.ventana_mm.notna().any():
        vm = res.ventana_mm.dropna()
        add("Tamaño de referencia: ventana medida", f"{_es(vm.min())}-{_es(vm.max())} mm",
            f"{VENTANA_MM:g} mm ± {TOL_VENTANA * 100:.0f} %",
            "OK" if (vm / VENTANA_MM - 1).abs().max() <= TOL_VENTANA else "REVISAR",
            "La regla y la plantilla no están en el mismo plano, o la ventana no se detectó entera.")
    rmin = (res.escala_foto_px_mm if "escala_foto_px_mm" in res else res.escala_px_mm).min()
    add("Resolución de la foto", f"{_es(rmin)} px/mm (la menor)", f"≥ {RES_OK:g} px/mm",
        "OK" if rmin >= RES_OK else ("REVISAR" if rmin >= RES_MIN else "NO VÁLIDO"),
        "Suba la foto original de la cámara (sin recortar, sin captura de pantalla, sin pasar por mensajería) "
        "y encuadre ventana y regla ocupando buena parte de la imagen.")
    if (res.aviso_escala).any():
        add("Escala: regla frente a ventana de 3 cm", "aviso en " + ", ".join(res.emplazamiento[res.aviso_escala]),
            f"regla y ventana coinciden ±{TOL_VENTANA * 100:.0f} %", "REVISAR",
            "Revise la figura de control: los puntos rojos deben caer sobre las marcas de milímetro.")
    if n > 1:
        for var, nom, accion in [
                ("escala_foto_px_mm", "CV de la escala de las fotos",
                 "La distancia de la cámara cambió entre fotos. La normalización a la resolución de análisis corrige el "
                 "recuento, pero el enfoque y la luz pueden variar: repítalas a la misma distancia."),
                ("area_alveolar_%", "CV del área alveolar", "Muestreo o luz desiguales entre rebanadas: revise las fotos."),
                ("densidad_alv_cm2", "CV de la densidad alveolar", "Muestreo o luz desiguales entre rebanadas: revise las fotos.")]:
            cv = _cv(res[var])
            add(nom, f"{_es(cv)} %", f"≤ {CV_MAX[var]:g} %", _estado_cv(cv, CV_MAX[var]), accion)
    nominal = (lado / 10) ** 2
    desv = (res.roi_cm2 / nominal - 1).abs().max()
    add("Área analizada", f"{_es(res.roi_cm2.min(), 2)}-{_es(res.roi_cm2.max(), 2)} cm²",
        f"{_es(nominal, 2)} cm² ± {ROI_TOL * 100:.0f} %", "OK" if desv <= ROI_TOL else "REVISAR",
        "La ventana de la plantilla no se detectó entera: revise la figura de control.")
    fuera = res[(res["area_alveolar_%"] < AREA_PLAUSIBLE[0]) | (res["area_alveolar_%"] > AREA_PLAUSIBLE[1])]
    add("Área alveolar plausible", f"{_es(res['area_alveolar_%'].min())}-{_es(res['area_alveolar_%'].max())} %",
        f"{AREA_PLAUSIBLE[0]:g}-{AREA_PLAUSIBLE[1]:g} %", "OK" if fuera.empty else "REVISAR",
        "Por debajo, el umbral pierde alvéolos; por encima, marca pared como alvéolo. Mire los contornos.")
    re_max = res.rel_ejes_mediana.max()
    add("Forma de los contornos (relación de ejes)", _es(re_max, 2), f"≤ {_es(REL_EJES_MAX)}",
        "OK" if not (re_max > REL_EJES_MAX) else "REVISAR",
        "Contornos alargados: grietas del corte o alvéolos mal separados. Mire los contornos.")
    p90 = res.feret_p90_mm.max(); lim = FERET_P90_MAX_FRAC * lado
    add("Feret P90 frente al tamaño de la zona", f"{_es(p90, 2)} mm", f"≤ {_es(lim)} mm",
        "OK" if not (p90 > lim) else "NO VÁLIDO",
        "Hay «alvéolos» del tamaño del campo: fusión masiva o fondo detectado como alvéolo.")
    cmin = res.contraste_DE.min()
    add("Contraste de la imagen", f"{_es(cmin)} niveles de gris (el menor)", f"≥ {CONTRASTE_BAJO:g}",
        "OK" if cmin >= CONTRASTE_BAJO else "REVISAR",
        "Miga poco contrastada: use luz más rasante para que cada alvéolo proyecte sombra.")
    return pd.DataFrame(filas)


def html_validacion(val, titulo, nota_final=True):
    """Tabla de semáforos legible en tema claro y oscuro."""
    estado = _peor(val.Estado.tolist())
    fondo, tinta, ic = ESTILO_ESTADO[estado]
    texto = {"OK": "Resultados válidos para interpretar.",
             "REVISAR": "Resultados utilizables con reservas: lea las acciones indicadas.",
             "PENDIENTE": "Resultados pendientes de validar con el blanco control.",
             "NO VÁLIDO": "Resultados no válidos: no los use en el informe sin corregir la causa."}[estado]
    filas = ""
    for r in val.itertuples():
        f, t, i = ESTILO_ESTADO[r.Estado]
        filas += (f"<tr><td style='padding:4px 8px;background:#ffffff;color:#1f2328'>{r.Control}</td>"
                  f"<td style='padding:4px 8px;background:#ffffff;color:#1f2328'>{r.Valor}</td>"
                  f"<td style='padding:4px 8px;background:#ffffff;color:#5f5e58'>{r.Criterio}</td>"
                  f"<td style='padding:4px 8px;background:{f};color:{t};font-weight:700;white-space:nowrap'>{i} {r.Estado}</td>"
                  f"<td style='padding:4px 8px;background:#ffffff;color:#5f5e58;font-size:12px'>{r.Acción}</td></tr>")
    cab = "".join(f"<th style='padding:5px 8px;background:#f4f3ef;color:#2b2b2b;text-align:left'>{h}</th>"
                  for h in ["Control", "Valor", "Criterio", "Estado", "Qué hacer"])
    extra = ("<div style='font-size:12px;color:#5f5e58;background:#ffffff;padding:6px 2px 0'>Control que no se "
             "automatiza: mire el <b>detalle ampliado</b> de la figura de segmentación. Los contornos deben caer "
             "sobre alvéolos, no sobre pared de miga.</div>") if nota_final else ""
    return (f"<div style='border:1px solid #d9d7cf;border-radius:8px;padding:10px 12px;margin:8px 0;background:#ffffff'>"
            f"<div style='font-size:15px;font-weight:700;color:#1f2328;margin-bottom:6px'>{titulo}</div>"
            f"<div style='background:{fondo};color:{tinta};border-radius:6px;padding:8px 12px;font-weight:700;"
            f"margin-bottom:8px'>{ic} {estado} · {texto}</div>"
            f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:13px;width:100%'>"
            f"<tr>{cab}</tr>{filas}</table></div>{extra}</div>"), estado


# ---------- Registro de la sesión (sobrevive a «Nuevo análisis») ----------
REGISTRO_DIR = "registro_sesion_PR3"
REG_REPLICAS = os.path.join(REGISTRO_DIR, "replicas.csv")
REG_BLANCO = os.path.join(REGISTRO_DIR, "blanco.json")
VARS_COMP = [("area_alveolar_%", "Área alveolar (%)"), ("densidad_alv_cm2", "Densidad (alv/cm²)"),
             ("d50_area_mm", "D50 por área (mm)"), ("feret_p90_mm", "Feret P90 (mm)"),
             ("%_area_gt2mm", "% del área en alvéolos > 2 mm")]
VAR_PRIMARIA = "area_alveolar_%"
# Distribución de tamaños por réplica: % acumulado del ÁREA alveolar en alvéolos con Feret ≤ x mm.
CDF_X = np.round(np.arange(0.1, 3.0001, 0.1), 1)
CDF_COLS = [f"cdf_area_{x:.1f}" for x in CDF_X]
CAMPOS_REG = (["muestra", "replica", "archivo", "fecha", "configuracion", "estado_sin_blanco", "escala_foto_px_mm",
               "escala_px_mm",
               "area_alveolar_%", "densidad_alv_cm2", "feret_mediano_mm", "feret_p90_mm", "d50_area_mm"]
              + [f"% {nom}" for nom in _nombres_clase()] + ["%_area_gt2mm", "contraste_DE", "controles_revisar"]
              + CDF_COLS)


def distribucion_area(alv_rep):
    """Curva acumulada del área alveolar frente al Feret y D50 por área (Feret que reparte el área
    alveolar en dos mitades). Describe la forma de toda la distribución, no solo un resumen."""
    if alv_rep is None or not len(alv_rep):
        return np.full(len(CDF_X), np.nan), np.nan
    f = alv_rep.feret_mm.values; a = alv_rep.area_mm2.values
    o = np.argsort(f); f, a = f[o], a[o]
    acum = 100 * np.cumsum(a) / a.sum()
    cdf = np.array([acum[f <= x][-1] if (f <= x).any() else 0.0 for x in CDF_X])
    d50 = float(np.interp(50, acum, f))
    return np.round(cdf, 2), round(d50, 3)


def _leer_csv(ruta):
    return pd.read_csv(ruta, sep=";", decimal=",", encoding="utf-8-sig")


def _escribir_csv(df, ruta):
    df.to_csv(ruta, index=False, sep=";", decimal=",", encoding="utf-8-sig")


TEXTO_REG = ("muestra", "replica", "archivo", "fecha", "configuracion", "estado_sin_blanco", "controles_revisar")
HOJA_REGISTRO = "_registro"   # hoja oculta del Excel de cada grupo: permite «Importar resultados anteriores»


def registro_a_excel(ruta_xlsx, df):
    """Guarda las filas del registro en una hoja oculta del Excel del grupo (sustituye al antiguo CSV)."""
    from openpyxl import load_workbook
    wb = load_workbook(ruta_xlsx)
    if HOJA_REGISTRO in wb.sheetnames:
        del wb[HOJA_REGISTRO]
    ws = wb.create_sheet(HOJA_REGISTRO); ws.sheet_state = "hidden"
    cols = list(df.columns); ws.append(cols)
    for fila in df.itertuples(index=False):
        ws.append([None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)
                   for v in fila])
    wb.save(ruta_xlsx)


def excel_simple(ruta_xlsx, hojas):
    """Excel sencillo con una hoja por tabla: {nombre: DataFrame}."""
    from openpyxl.styles import Font, PatternFill
    with pd.ExcelWriter(ruta_xlsx, engine="openpyxl") as w:
        for nombre, df in hojas.items():
            df.to_excel(w, sheet_name=nombre[:31], index=False)
            ws = w.sheets[nombre[:31]]
            for c in ws[1]:
                c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="D9D9D9")
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(48, max(10, max(len(str(c.value or "")) for c in col) + 2))


def _tipos_registro(reg):
    for c in CAMPOS_REG:
        if c not in reg.columns:          # registros de versiones anteriores
            reg[c] = np.nan
    for c in CAMPOS_REG:
        if c not in TEXTO_REG and c in reg.columns:
            reg[c] = pd.to_numeric(reg[c], errors="coerce")
    return reg


def registro_cargar():
    try:
        return _tipos_registro(_leer_csv(REG_REPLICAS))
    except Exception:
        return pd.DataFrame(columns=CAMPOS_REG)


def registro_guardar(nuevas):
    """Añade las réplicas de una pieza; si la pieza ya estaba, la sustituye."""
    os.makedirs(REGISTRO_DIR, exist_ok=True)
    reg = registro_cargar()
    reg = reg[~reg.muestra.isin(nuevas.muestra.unique())]
    reg = _tipos_registro(pd.concat([reg, nuevas[CAMPOS_REG]], ignore_index=True) if len(reg) else
                          nuevas[CAMPOS_REG].copy())
    _escribir_csv(reg, REG_REPLICAS)
    return reg


def filas_registro(muestra, res, estado, opciones, fecha, alv=None, revisar=""):
    d = res.rename(columns={"emplazamiento": "replica"}).copy()
    d["muestra"] = muestra
    for i, rep in enumerate(d.replica):
        cdf, d50 = distribucion_area(alv[alv.emplazamiento == rep] if alv is not None else None)
        d.loc[d.index[i], "d50_area_mm"] = d50
        for c, v in zip(CDF_COLS, cdf):
            d.loc[d.index[i], c] = v
    d["fecha"] = f"{fecha:%d/%m/%Y %H:%M}"; d["configuracion"] = configuracion(opciones)
    d["estado_sin_blanco"] = estado; d["controles_revisar"] = revisar
    for c in CAMPOS_REG:
        if c not in d.columns:
            d[c] = np.nan
    return d[CAMPOS_REG]


def registro_importar(rutas):
    """Incorpora al registro réplicas de análisis anteriores: hoja oculta «_registro» del Excel de cada grupo
    (suelto o dentro del ZIP de resultados). Lee también los «_registro_replicas.csv» de versiones anteriores.
    Devuelve (piezas añadidas, errores)."""
    import io as _io
    añadidas, errores = [], []
    for ruta in rutas:
        try:
            tablas = []
            if ruta.lower().endswith(".zip"):
                with zipfile.ZipFile(ruta) as z:
                    for nom in z.namelist():
                        if nom.endswith(".xlsx"):
                            try:
                                tablas.append(pd.read_excel(_io.BytesIO(z.read(nom)), sheet_name=HOJA_REGISTRO))
                            except Exception:
                                pass
                        elif nom.endswith("_registro_replicas.csv"):
                            tablas.append(pd.read_csv(_io.BytesIO(z.read(nom)), sep=";", decimal=",",
                                                      encoding="utf-8-sig"))
            elif ruta.lower().endswith(".xlsx"):
                try:
                    tablas.append(pd.read_excel(ruta, sheet_name=HOJA_REGISTRO))
                except Exception:
                    pass
            else:
                tablas.append(_leer_csv(ruta))
            if not tablas:
                errores.append(f"{os.path.basename(ruta)}: no contiene resultados de grupos del aplicativo "
                               "(se importa el ZIP de resultados o el Excel «_resultados.xlsx» de un grupo)")
            for t in tablas:
                falta = [c for c in ("muestra", "replica", "area_alveolar_%", "estado_sin_blanco") if c not in t.columns]
                if falta:
                    errores.append(f"{os.path.basename(ruta)}: faltan columnas {', '.join(falta)}"); continue
                for c in CAMPOS_REG:
                    if c not in t.columns:
                        t[c] = np.nan
                registro_guardar(t); añadidas += t.muestra.unique().tolist()
        except Exception as e:
            errores.append(f"{os.path.basename(ruta)}: {e}")
    return añadidas, errores


def blanco_cargar():
    try:
        with open(REG_BLANCO, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def blanco_guardar(d):
    os.makedirs(REGISTRO_DIR, exist_ok=True)
    with open(REG_BLANCO, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


# ---------- Comparación entre grupos ----------
# Dos criterios complementarios, con la réplica como unidad (nunca el alvéolo: los alvéolos de una
# misma foto no son independientes entre sí):
#   1) Regla del guion: dos grupos difieren si los recorridos (mín-máx) de sus réplicas no se
#      solapan. Con 3 frente a 3 réplicas equivale a la prueba exacta de Mann-Whitney con U = 0:
#      p = 0,10 bilateral, el valor más bajo que 3 réplicas pueden dar. No corrige por comparaciones
#      múltiples: con k grupos hay k(k-1)/2 pares y, si no hubiera ninguna diferencia real, cabría
#      esperar un 10 % de «distintas» por azar.
#   2) ANOVA de un factor + Tukey HSD: usa la variabilidad de todas las réplicas a la vez (más grados
#      de libertad) y controla el error del conjunto de comparaciones (α = 0,05). Supone normalidad
#      y varianzas parecidas, que con 3 réplicas no se pueden comprobar.
ALFA = 0.05
P_RECORRIDOS = 0.10          # p exacta de la regla de recorridos con 3 frente a 3 réplicas
PALETA_GRUPOS = ["#2a78d6", "#e8702a", "#2e9e5b", "#c23b5a", "#7a4fc2", "#8a6d1f", "#1aa3a3",
                 "#d1569c", "#5b6b7a", "#a33b1f", "#3f7f2a", "#4060a8"]


def _tukey(muestras):
    """Matriz de p ajustados de Tukey HSD, o None si no se puede calcular."""
    try:
        from scipy.stats import tukey_hsd
        if len(muestras) < 2 or any(len(m) < 2 for m in muestras):
            return None
        if sum(len(m) for m in muestras) - len(muestras) < 2:
            return None
        if all(np.ptp(m) == 0 for m in muestras):
            return None
        return np.asarray(tukey_hsd(*muestras).pvalue)
    except Exception:
        return None


def _anova(muestras):
    try:
        from scipy.stats import f_oneway
        return float(f_oneway(*muestras).pvalue)
    except Exception:
        return np.nan


def letras_grupo(medias, distintos):
    """Letras de agrupación (algoritmo de inserción y absorción). Dos grupos que comparten letra no
    se distinguen. medias: {grupo: media}; distintos: conjunto de pares (a, b) significativos."""
    orden = sorted(medias, key=lambda g: -medias[g])
    conjuntos = [set(orden)]
    for a, b in itertools.combinations(orden, 2):
        if (a, b) not in distintos and (b, a) not in distintos:
            continue
        nuevos = []
        for s in conjuntos:
            if a in s and b in s:
                nuevos += [s - {a}, s - {b}]
            else:
                nuevos.append(s)
        nuevos = [s for s in nuevos if s]
        conjuntos = [s for i, s in enumerate(nuevos)
                     if not any((s < t) or (s == t and j < i) for j, t in enumerate(nuevos) if j != i)]
    conjuntos.sort(key=lambda s: min(orden.index(g) for g in s))
    letras = {g: "" for g in orden}
    for i, s in enumerate(conjuntos):
        for g in s:
            letras[g] += "abcdefghijklmnopqrstuvwxyz"[i % 26]
    return letras


def comparar_grupos(reg, blanco, grupos=None, ruta_base=None):
    eb = blanco["estado"] if blanco else "PENDIENTE"
    grupos = [g for g in dict.fromkeys(reg.muestra) if grupos is None or g in grupos]
    reg = reg[reg.muestra.isin(grupos)]
    estado = {g: _peor([reg[reg.muestra == g].estado_sin_blanco.iloc[0], eb]) for g in grupos}
    usables = [g for g in grupos if estado[g] != "NO VÁLIDO"]
    datos = {g: {v: reg[reg.muestra == g][v].astype(float).dropna().values for v, _ in VARS_COMP} for g in grupos}

    # Resumen por grupo
    filas = []
    for g in grupos:
        f = {"Grupo": g, "n réplicas": int((reg.muestra == g).sum()), "Estado": estado[g]}
        for v, nom in VARS_COMP:
            x = datos[g][v]
            f[f"{nom} · media"] = round(x.mean(), 2) if len(x) else np.nan
            f[f"{nom} · DE"] = round(x.std(ddof=1), 2) if len(x) > 1 else np.nan
            f[f"{nom} · mín"] = round(x.min(), 2) if len(x) else np.nan
            f[f"{nom} · máx"] = round(x.max(), 2) if len(x) else np.nan
        filas.append(f)
    resumen = pd.DataFrame(filas)

    # Pruebas por variable
    tukey, anova, letras = {}, {}, {}
    for v, nom in VARS_COMP:
        muestras = [datos[g][v] for g in usables if len(datos[g][v])]
        gs = [g for g in usables if len(datos[g][v])]
        P = _tukey(muestras)
        tukey[v] = {(a, b): P[i, j] for i, a in enumerate(gs) for j, b in enumerate(gs) if i != j} if P is not None else {}
        anova[v] = _anova(muestras) if len(muestras) >= 2 else np.nan
        dist = {(a, b) for (a, b), p in tukey[v].items() if p < ALFA}
        letras[v] = letras_grupo({g: datos[g][v].mean() for g in gs}, dist) if tukey[v] else {}
    for v, nom in VARS_COMP:
        resumen[f"{nom} · letras"] = [letras[v].get(g, "") for g in grupos]

    # Pares
    pares = []
    for a, b in itertools.combinations(grupos, 2):
        for v, nom in VARS_COMP:
            xa, xb = datos[a][v], datos[b][v]
            if not len(xa) or not len(xb):
                continue
            solape = xa.min() <= xb.max() and xb.min() <= xa.max()
            p = tukey[v].get((a, b), np.nan)
            ea, eb_ = estado[a], estado[b]
            if "NO VÁLIDO" in (ea, eb_):
                lectura = "No interpretable: algún grupo no es válido"
            elif "PENDIENTE" in (ea, eb_):
                lectura = "Pendiente: falta validar el blanco control"
            else:
                sig = (not np.isnan(p)) and p < ALFA
                if not solape and sig:
                    lectura = "Diferencia sólida (los dos criterios)"
                elif not solape or sig:
                    lectura = "Indicio de diferencia (un solo criterio)"
                else:
                    lectura = "No se puede afirmar que difieran"
                if "REVISAR" in (ea, eb_) and not lectura.startswith("No se"):
                    lectura += " · con reservas"
            pares.append({"Grupo A": a, "Grupo B": b, "Variable": nom,
                          "Media A": round(xa.mean(), 2), "Media B": round(xb.mean(), 2),
                          "Diferencia A − B": round(xa.mean() - xb.mean(), 2),
                          "Recorridos": "solapados" if solape else "sin solape",
                          "Tukey p ajustado": round(p, 3) if not np.isnan(p) else np.nan,
                          "Lectura": lectura})
    pares = pd.DataFrame(pares)
    n_pares = len(grupos) * (len(grupos) - 1) // 2
    info = dict(n_grupos=len(grupos), n_pares=n_pares, falsos_esperados=round(P_RECORRIDOS * n_pares, 1),
                anova=anova, tukey_ok=any(tukey.values()))

    figs = {"variables": figura_variables(reg, grupos, datos, estado, letras),
            "distribucion": figura_distribucion(reg, grupos, estado),
            "matriz": figura_matriz(grupos, datos, estado, tukey)}
    archivos = []
    if ruta_base:
        excel_simple(ruta_base + "_resultados.xlsx", {"Pares": pares, "Resumen por grupo": resumen})
        archivos = [ruta_base + "_resultados.xlsx"]
        for k, f in figs.items():
            if f is not None:
                f.savefig(ruta_base + f"_figura_{k}.png", dpi=140, facecolor="white"); archivos.append(ruta_base + f"_figura_{k}.png")
    configs = reg.configuracion.dropna().unique()
    res_g = reg.groupby("muestra").escala_px_mm.median().dropna()
    info["resoluciones"] = res_g.round(1).to_dict()
    info["resolucion_desigual"] = bool(len(res_g) > 1 and res_g.max() / res_g.min() - 1 > 0.05)
    return dict(resumen=resumen, pares=pares, figs=figs, archivos=archivos, configs=configs, info=info)


def _color_estado(e, base):
    return "#a1251f" if e == "NO VÁLIDO" else ("#9a988f" if e == "PENDIENTE" else base)


def figura_variables(reg, grupos, datos, estado, letras):
    """Una gráfica por variable. Orden fijo por el área alveolar (de mayor a menor) para leer todas
    igual. Puntos = réplicas; barra = recorrido; raya = media; letras = Tukey HSD."""
    orden = sorted(grupos, key=lambda g: -(datos[g][VAR_PRIMARIA].mean() if len(datos[g][VAR_PRIMARIA]) else -1e9))
    nv = len(VARS_COMP); nc = 2; nf = int(np.ceil(nv / nc))
    ancho = max(11, 1.1 * len(orden) * nc + 3)
    fig, axs = plt.subplots(nf, nc, figsize=(ancho, 4.3 * nf), squeeze=False)
    x = np.arange(len(orden)); rot = 30 if len(orden) > 4 or max(len(g) for g in orden) > 8 else 0
    for k, (a, (v, nom)) in enumerate(zip(axs.flat, VARS_COMP)):
        ymax = max((datos[g][v].max() for g in orden if len(datos[g][v])), default=1)
        ymin = min((datos[g][v].min() for g in orden if len(datos[g][v])), default=0)
        for i, g in enumerate(orden):
            y = datos[g][v]
            if not len(y):
                continue
            c = _color_estado(estado[g], "#2a78d6")
            a.vlines(i, y.min(), y.max(), color=c, lw=9, alpha=0.25)
            a.scatter(np.full(len(y), i) + np.linspace(-0.06, 0.06, len(y)), y, color=c, s=34, zorder=3)
            a.hlines(y.mean(), i - 0.22, i + 0.22, color=TINTA, lw=1.8, zorder=4)
            if letras[v].get(g):
                a.text(i, ymax + 0.06 * (ymax - ymin + 1e-9), letras[v][g], ha="center", va="bottom",
                       fontsize=11, weight="bold", color=TINTA)
        a.set_xticks(x); a.set_xticklabels(orden, rotation=rot, ha="right" if rot else "center", color=TINTA)
        a.set_xlim(-0.6, len(orden) - 0.4); a.set_ylim(top=ymax + 0.2 * (ymax - ymin + 1e-9))
        a.set_title(nom + (" · variable primaria" if v == VAR_PRIMARIA else ""), loc="left", fontsize=11.5,
                    color=TINTA, weight="bold" if v == VAR_PRIMARIA else "normal")
        _limpiar(a)
    for a in list(axs.flat)[nv:]:
        a.axis("off")
    fig.suptitle("Comparación entre grupos · puntos = réplicas, barra = recorrido mín-máx, raya = media.\n"
                 "Letras: grupos que comparten letra no difieren (Tukey HSD, α = 0,05). Gris = pendiente de blanco; "
                 "rojo = no válido.", fontsize=11, x=0.01, ha="left", y=0.995, color=TINTA)
    plt.tight_layout(rect=[0, 0, 1, 1 - 0.42 / (4.3 * nf)])
    return fig


def figura_distribucion(reg, grupos, estado):
    """Curva acumulada del área alveolar frente al tamaño (Feret), por grupo: línea = mediana de las
    réplicas, banda = mín-máx. Si las bandas de dos grupos se separan en un tramo, sus migas difieren
    en ese rango de tamaños. Desplazada a la derecha = el área está en alvéolos más grandes."""
    if not all(c in reg.columns for c in CDF_COLS) or reg[CDF_COLS].isna().all().all():
        return None
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    a = ax[0]
    for i, g in enumerate(grupos):
        s = reg[reg.muestra == g][CDF_COLS].astype(float).dropna()
        if s.empty:
            continue
        c = _color_estado(estado[g], PALETA_GRUPOS[i % len(PALETA_GRUPOS)])
        med = s.median().values
        a.fill_between(CDF_X, s.min().values, s.max().values, color=c, alpha=0.18, lw=0)
        d50 = reg[reg.muestra == g].d50_area_mm.astype(float)
        a.plot(CDF_X, med, color=c, lw=2, label=f"{g} · D50 (mediana) {_es(d50.median(), 2)} mm")
    a.axhline(50, color=GRIS, lw=1, ls=":")
    for xl in _limites_clase():
        a.axvline(xl, color=GRIS, lw=1, ls="--")
    a.set_xlim(CDF_X[0], CDF_X[-1]); a.set_ylim(0, 101)
    a.set_xlabel("diámetro de Feret (mm)", color=GRIS); a.set_ylabel("% acumulado del área alveolar", color=GRIS)
    a.set_title("Distribución del área alveolar por tamaño (línea = mediana, banda = réplicas)", loc="left",
                fontsize=11.5, color=TINTA)
    a.legend(fontsize=9, frameon=False, loc="lower right"); _limpiar(a)
    # D50 por réplica
    b = ax[1]
    orden = sorted(grupos, key=lambda g: reg[reg.muestra == g].d50_area_mm.astype(float).median())
    for i, g in enumerate(orden):
        y = reg[reg.muestra == g].d50_area_mm.astype(float).dropna().values
        if not len(y):
            continue
        c = _color_estado(estado[g], PALETA_GRUPOS[grupos.index(g) % len(PALETA_GRUPOS)])
        b.hlines(i, y.min(), y.max(), color=c, lw=9, alpha=0.25)
        b.scatter(y, np.full(len(y), i), color=c, s=34, zorder=3)
        b.vlines(y.mean(), i - 0.25, i + 0.25, color=TINTA, lw=1.8, zorder=4)
    b.set_yticks(range(len(orden))); b.set_yticklabels(orden, color=TINTA)
    b.set_xlabel("D50 por área (mm)", color=GRIS)
    b.set_title("D50: tamaño que reparte el área alveolar en dos mitades", loc="left", fontsize=11.5, color=TINTA)
    _limpiar(b); b.grid(axis="x", color="#ecebe6", lw=0.8)
    plt.tight_layout()
    return fig


def figura_matriz(grupos, datos, estado, tukey):
    """Matriz de pares para la variable primaria: color = p ajustado de Tukey; cifra = diferencia de
    medias (fila − columna) en puntos de área alveolar; ≠ = recorridos sin solape."""
    v = VAR_PRIMARIA; n = len(grupos)
    if n < 2:
        return None
    from matplotlib.colors import ListedColormap, BoundaryNorm
    M = np.full((n, n), np.nan)
    for i, a in enumerate(grupos):
        for j, b in enumerate(grupos):
            if i > j and (a, b) in tukey[v]:
                p = tukey[v][(a, b)]
                M[i, j] = 0 if p < 0.01 else (1 if p < ALFA else 2)
    cmap = ListedColormap(["#1b6b3a", "#7cc49a", "#e6e4dd"]); cmap.set_bad("#ffffff")
    lado = max(4.5, 0.95 * (n - 1) + 2.2)
    fig, a = plt.subplots(figsize=(lado + 2, lado))
    a.imshow(np.ma.masked_invalid(M[1:, :-1]), cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3))
    for i, ga in enumerate(grupos):
        for j, gb in enumerate(grupos):
            if i <= j:
                continue
            xa, xb = datos[ga][v], datos[gb][v]
            if not len(xa) or not len(xb):
                continue
            sin_solape = not (xa.min() <= xb.max() and xb.min() <= xa.max())
            txt = f"{xa.mean() - xb.mean():+.1f}".replace(".", ",") + (" ≠" if sin_solape else "")
            if "NO VÁLIDO" in (estado[ga], estado[gb]):
                txt = "no válido"
            a.text(j, i - 1, txt, ha="center", va="center", fontsize=10,
                   color="white" if M[i, j] == 0 else TINTA, weight="bold" if sin_solape else "normal")
    a.set_xticks(range(n - 1)); a.set_xticklabels(grupos[:-1], rotation=30, ha="right", color=TINTA)
    a.set_yticks(range(n - 1)); a.set_yticklabels(grupos[1:], color=TINTA)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_xticks(np.arange(-0.5, n - 1), minor=True); a.set_yticks(np.arange(-0.5, n - 1), minor=True)
    a.grid(which="minor", color="white", lw=2); a.tick_params(which="minor", length=0)
    a.legend(handles=[Patch(color="#1b6b3a", label="p < 0,01"), Patch(color="#7cc49a", label="p < 0,05"),
                      Patch(color="#e6e4dd", label="p ≥ 0,05 (sin diferencia)")],
             loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9, title="Tukey HSD",
             title_fontsize=9)
    a.set_title("Área alveolar: diferencia de medias (fila − columna, puntos %)\n≠ = recorridos de las réplicas "
                "sin solape", loc="left", fontsize=11, color=TINTA)
    plt.tight_layout()
    return fig


def html_tabla(df, resaltar=(), col_estado=None):
    cab = "".join(f"<th style='padding:4px 8px;background:#f4f3ef;color:#2b2b2b;text-align:center'>{c}</th>"
                  for c in df.columns)
    filas = ""
    for _, r in df.iterrows():
        celdas = ""
        for c in df.columns:
            v = r[c]
            txt = _es(v, 2) if isinstance(v, (float, np.floating)) else str(v)
            estilo = "background:#ffffff;color:#1f2328"
            if c == col_estado and v in ESTILO_ESTADO:
                f, t, i = ESTILO_ESTADO[v]; estilo = f"background:{f};color:{t};font-weight:700"; txt = f"{i} {v}"
            elif isinstance(v, str) and v.startswith(("Distintas", "Diferencia sólida")):
                estilo = "background:#e9f6ee;color:#1b6b3a;font-weight:600"
            elif isinstance(v, str) and v.startswith("Indicio"):
                estilo = "background:#fff6e5;color:#8a5300;font-weight:600"
            celdas += f"<td style='padding:4px 8px;{estilo}'>{txt}</td>"
        filas += f"<tr>{celdas}</tr>"
    return (f"<div style='overflow-x:auto;background:#ffffff'><table style='border-collapse:collapse;"
            f"font-size:13px;border:1px solid #d9d7cf'><tr>{cab}</tr>{filas}</table></div>")



# ============================================================
#  ANÁLISIS DE UNA PIEZA Y DEL BLANCO CONTROL
# ============================================================
def _mostrar_tabla(tabla_r):
    from IPython.display import display
    try:
        fmt = lambda v: (f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                         if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else ("" if pd.isna(v) else v))
        sty = (tabla_r.set_index("Réplica").style.format(fmt)
               .apply(lambda r: ["background:#eef4fc;color:#1f2328;font-weight:600" if r.name in FILAS_RESUMEN else ""
                                 for _ in r], axis=1)
               .set_table_styles([{"selector": "th", "props": "background:#f4f3ef;color:#2b2b2b;font-weight:600;"
                                                          "padding:4px 8px;text-align:center"},
                                  {"selector": "td", "props": "padding:4px 8px;text-align:right;background:#ffffff;color:#1f2328"},
                                  {"selector": "", "props": "border-collapse:collapse;background:#ffffff"}]))
        display(sty)
    except Exception:
        display(tabla_r.set_index("Réplica"))


def _zip(base, extra=()):
    zipname = base + "_resultados.zip"
    with zipfile.ZipFile(zipname, "w") as z:
        for f in sorted(set(glob.glob(base + "_*")) | set(extra)):
            if not f.endswith(".zip") and os.path.exists(f):
                z.write(f, os.path.basename(f))
    return zipname


def analizar_muestra(rutas, muestra, opciones, salida="resultados", mostrar_control="Solo si hay avisos",
                     al_avanzar=None, etiquetas=None):
    """Analiza las réplicas de una pieza, la valida y la guarda en el registro de la sesión."""
    import datetime
    from IPython.display import display, HTML
    os.makedirs(salida, exist_ok=True)
    base = os.path.join(salida, _archivo_seguro(muestra))
    reps = list(etiquetas) if etiquetas else [f"Réplica {i}" for i in range(1, len(rutas) + 1)]
    resumen, todos, paneles, avisos = [], [], [], []
    for i_rep, (etq, ruta) in enumerate(zip(reps, rutas)):
        if al_avanzar:
            al_avanzar(i_rep, len(rutas), etq)
        r, df, panel, txt = analizar_foto(ruta, etq, opciones, base, mostrar_control)
        avisos.append(txt)
        if r is None:
            continue
        resumen.append({"muestra": muestra, **r}); todos.append(df); paneles.append(panel)
    for a in avisos:
        _html(a, "aviso" if "⚠" in a else "ok")
    if not resumen:
        raise RuntimeError("no se pudo medir ninguna foto. Lea los avisos en amarillo: indican la causa y cómo corregirla.")
    res = pd.DataFrame(resumen); alv = pd.concat(todos, ignore_index=True)
    if al_avanzar:
        al_avanzar(len(rutas), len(rutas), "fin")

    # Figuras
    figura(paneles, f"{muestra}: análisis alveolar por escala de grises", opciones["lado"],
           base + "_figura_segmentacion.png")
    plt.show()
    dg = datos_graficos(res, alv)
    panel_graficos(res, alv, f"{muestra}: densidad, distribución y área alveolar", base + "_graficos.png", dg)
    plt.show()

    # Tabla de cálculos
    tabla_r = tabla_calculos(res)
    _html("<b>Cálculos</b> · área alveolar = Σ área de alvéolos / área analizada × 100 (variable primaria) · "
          "densidad = N / área analizada (cm²) · % clase = n clase / N × 100 · "
          f"% del área en alvéolos > {_es(LIM_MUY_GRANDES_MM, 0)} mm = Σ área de alvéolos con Feret > "
          f"{_es(LIM_MUY_GRANDES_MM, 0)} mm / Σ área de alvéolos × 100 · CV = DE / media × 100 · "
          "recorrido = mín-máx de las réplicas.<br>N se cuenta con marco insesgado (sin los alvéolos que tocan el "
          f"borde derecho o inferior); mínimo contado: {_es(opciones['dmin'], 2)} mm de diámetro equivalente; "
          f"umbral: {opciones['umbral']} con contraste mínimo de {CONTRASTE_MIN:g} niveles de gris.", "info")
    _mostrar_tabla(tabla_r)

    # Validación: réplicas + blanco de la sesión
    val_sin = validar_muestra(res, opciones, len(rutas))
    estado_sin = _peor(val_sin.Estado.tolist())
    blanco = blanco_cargar()
    val = pd.concat([pd.DataFrame([fila_blanco(blanco)]), val_sin], ignore_index=True)
    html, estado = html_validacion(val, f"Validación de «{muestra}»")
    display(HTML(html))

    # Registro de la sesión y archivos
    fecha = datetime.datetime.now()
    revisar = "; ".join(val_sin.loc[val_sin.Estado.isin(["REVISAR", "NO VÁLIDO"]), "Control"].astype(str))
    nuevas = filas_registro(muestra, res, estado_sin, opciones, fecha, alv, revisar=revisar)
    registro_guardar(nuevas)
    alv_out = alv.drop(columns=["k", "etiqueta"]).rename(columns={
        "emplazamiento": "Réplica", "area_mm2": "Área (mm²)", "deq_mm": "D equivalente (mm)",
        "feret_mm": "Feret (mm)", "rel_ejes": "Relación de ejes", "clase": "Clase",
        "borde_insesgado": "Toca borde derecho/inferior (no se cuenta en N)"})
    hechos, fallos = generar_informes(base, muestra, tabla_r, res, alv, alv_out, dg, opciones, avisos, val)
    # Sin CSV: el registro del grupo viaja en una hoja oculta del Excel (para «Importar resultados anteriores»)
    try:
        registro_a_excel(base + "_resultados.xlsx", nuevas)
    except Exception as e:
        fallos.append(f"registro dentro del Excel ({e})")
    try:
        reporte_hoja_d(base + "_reporte_Hoja_D_miga.docx", registro_cargar(), blanco, [muestra], comparar=False)
        hechos.append("Reporte")
    except Exception as e:
        fallos.append(f"reporte Hoja D ({e})")
    if hechos:
        _html("Documentos generados: " + ", ".join(
            {"Excel": "libro Excel con tablas, validación y gráficos", "Word": "ficha técnica en Word (editable)",
             "PDF": "ficha técnica en PDF", "Reporte": "reporte para la Hoja D · Miga (Word)"}[h] for h in hechos) + ". Se incluyen en el ZIP.", "ok")
    if fallos:
        _html("No se pudo generar: " + "; ".join(fallos) + ".", "aviso")
    return tabla_r, _zip(base), estado


def analizar_blanco(rutas, opciones, salida="resultados", mostrar_control="Solo si hay avisos",
                    al_avanzar=None, etiquetas=None):
    """Blanco control: cartulina gris mate fotografiada con la misma plantilla, luz y distancia.
    Una superficie sin alvéolos debe dar un área alveolar próxima a cero."""
    import datetime
    from IPython.display import display, HTML
    os.makedirs(salida, exist_ok=True)
    base = os.path.join(salida, "Blanco_control")
    reps = list(etiquetas) if etiquetas else [f"Blanco_{i}" for i in range(1, len(rutas) + 1)]
    resumen, paneles, avisos = [], [], []
    for i_rep, (etq, ruta) in enumerate(zip(reps, rutas)):
        if al_avanzar:
            al_avanzar(i_rep, len(rutas), etq)
        r, df, panel, txt = analizar_foto(ruta, etq, opciones, base, mostrar_control)
        avisos.append(txt)
        if r is not None:
            resumen.append(r); paneles.append(panel)
    for a in avisos:
        _html(a, "aviso" if "⚠" in a else "ok")
    if not resumen:
        raise RuntimeError("no se pudo medir la foto del blanco. Lea los avisos en amarillo: indican la causa y cómo corregirla.")
    if al_avanzar:
        al_avanzar(len(rutas), len(rutas), "fin")
    res = pd.DataFrame(resumen)
    figura(paneles, "Blanco control: debe verse sin contornos", opciones["lado"], base + "_figura_segmentacion.png")
    plt.show()
    peor = res.loc[res["area_alveolar_%"].idxmax()]
    area = float(peor["area_alveolar_%"])
    est, motivo = estado_blanco(area, float(res["contraste_DE"].min()), float(res["escala_foto_px_mm"].min()))
    fecha = datetime.datetime.now()
    blanco = {"fecha": f"{fecha:%d/%m/%Y %H:%M}", "archivos": "; ".join(res.archivo),
              "area_alveolar_%": area, "N": int(peor["N"]), "contraste_DE": float(peor["contraste_DE"]),
              "escala_px_mm": float(peor["escala_foto_px_mm"]), "estado": est, "motivo": motivo,
              "configuracion": configuracion(opciones)}
    blanco_guardar(blanco)
    # Mismas columnas que la hoja de registro (Hoja D · Miga) y que la tabla de las piezas,
    # seguidas de los controles propios del blanco.
    nc = _nombres_clase()
    cols = {"emplazamiento": "Foto", "archivo": "Archivo", "escala_foto_px_mm": "Escala de la foto (px/mm)",
            "area_alveolar_%": "Área alveolar (%)", "densidad_alv_cm2": "Densidad (alv/cm²)",
            "feret_p90_mm": "Feret P90 (mm)", f"% {nc[0]}": f"% {nc[0][0].lower() + nc[0][1:]}",
            f"% {nc[1]}": f"% {nc[1][0].lower() + nc[1][1:]}", f"% {nc[2]}": f"% {nc[2][0].lower() + nc[2][1:]}",
            "N": "N «alvéolos»", "contraste_DE": "Contraste local (DE)", "umbral_auto": "Umbral automático",
            "descartado_contraste_%": "Área descartada por contraste (%)"}
    tabla = res[[c for c in cols if c in res.columns]].rename(columns=cols)
    blanco.update({"feret_p90_mm": float(peor["feret_p90_mm"]) if pd.notna(peor["feret_p90_mm"]) else None,
                   "densidad_alv_cm2": float(peor["densidad_alv_cm2"])})
    for k in range(3):
        v = peor[f"% {nc[k]}"]
        blanco[f"% {nc[k]}"] = float(v) if pd.notna(v) else None
    blanco_guardar(blanco)
    if tabla["N «alvéolos»"].max() > 0:
        _html("En un blanco, los «alvéolos» son manchas o relieves de la cartulina. Su Feret P90 y su reparto por "
              "tamaños se anotan solo como control: lo que decide el veredicto es el área alveolar.", "info")
    display(HTML(html_tabla(tabla)))
    val = pd.DataFrame([fila_blanco(blanco)])
    html, _ = html_validacion(val, "Validación del blanco control", nota_final=False)
    display(HTML(html))
    _html("El blanco queda registrado para esta sesión y se aplica a todas las piezas, también a las ya "
          "analizadas (véase el paso 4). Si cambia la luz o la distancia de la cámara, repita el blanco.", "info")
    try:
        excel_simple(base + "_resultados.xlsx", {"Blanco": tabla, "Validacion": val})
    except Exception as e:
        _html(f"No se pudo generar el Excel del blanco ({e}).", "aviso")
    return blanco, _zip(base)


# ============================================================
#  INFORMES: gráficos, Excel con gráficos nativos, ficha técnica (DOCX y PDF)
# ============================================================
import io, re, base64, datetime, subprocess, sys
import matplotlib

LOGO_B64 = "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"

FICHA_TECNICA = False   # True: versión del profesor (ficha técnica en Word y PDF)
ROJO, TINTA_D, SUAVE, AZUL_D = "C00000", "1B1B1B", "555555", "2E5F8A"
AUTORIA = ("Prof. ", "Kerly Palacios")
DEPARTAMENTO = "Departamento de Biotecnología y Ciencias de los Alimentos"
NOMBRES_CLASE = _nombres_clase()
CLASES_NOMBRES_RES = NOMBRES_CLASE
HEX_CLASE = [c.lstrip("#").upper() for c in COLOR_CLASE]
TINTA, GRIS = "#2b2b2b", "#77756e"


def _asegurar(paquete, modulo):
    """Importa un módulo; si falta (Colab), lo instala con pip."""
    try:
        __import__(modulo); return True
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", paquete], check=True,
                           capture_output=True)
            __import__(modulo); return True
        except Exception:
            return False


def _logo():
    return io.BytesIO(base64.b64decode(LOGO_B64))


def _c(v, d=1):
    """Número con coma decimal."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _limites_clase():
    return [a for a, _, _ in CLASES[1:]]


# ---------- Datos base de los gráficos (una sola fuente para PNG, Excel y ficha) ----------
def datos_graficos(res, alv):
    reps = res.emplazamiento.tolist()
    cnt = alv[~alv.borde_insesgado.astype(bool)]
    dens = pd.DataFrame({"Réplica": reps, "Densidad (alv/cm²)": res.densidad_alv_cm2.round(2).values})
    dens["Media (alv/cm²)"] = round(res.densidad_alv_cm2.mean(), 2)
    F = cnt.feret_mm.values
    paso = HIST_PASO_MM
    lim_sup = _limites_clase()[-1] * 2.5          # el histograma llega al menos a 2,5 veces el límite de «grandes»
    ini = np.floor(F.min() / paso) * paso if len(F) else 0.0
    tope = max(lim_sup, np.ceil(np.quantile(F, 0.995) / paso) * paso + paso) if len(F) else lim_sup
    bordes = np.round(np.arange(ini, tope + paso / 2, paso), 4)
    h, _ = np.histogram(np.clip(F, bordes[0], bordes[-1] - 1e-6), bordes)   # el último intervalo acumula la cola
    pct = 100 * h / max(len(F), 1)
    cen = (bordes[:-1] + bordes[1:]) / 2
    filas = []
    for a, b, p, c_ in zip(bordes[:-1], bordes[1:], pct, cen):
        k = _clase(c_)
        fila = {"Intervalo de Feret (mm)": f"{_c(a)}-{_c(b)}", "Centro (mm)": round(c_, 3),
                "% de alvéolos": round(p, 2), "Clase": NOMBRES_CLASE[k]}
        for j, nom in enumerate(NOMBRES_CLASE):
            fila[nom] = round(p, 2) if j == k else 0.0
        filas.append(fila)
    dist = pd.DataFrame(filas)
    area = []
    for r in reps:
        s = cnt[cnt.emplazamiento == r]; tot = max(s.area_mm2.sum(), 1e-9)
        fila = {"Réplica": r}
        for k, nom in enumerate(NOMBRES_CLASE):
            fila[f"% área {_corto(nom)}"] = round(100 * s[s.k == k].area_mm2.sum() / tot, 2)
        fila["Área alveolar (% de la superficie)"] = float(res.loc[res.emplazamiento == r, "area_alveolar_%"].iloc[0])
        area.append(fila)
    area = pd.DataFrame(area)
    tot = max(cnt.area_mm2.sum(), 1e-9)
    global_num = [100 * (cnt.k == k).mean() if len(cnt) else 0 for k in range(3)]
    global_area = [100 * cnt[cnt.k == k].area_mm2.sum() / tot for k in range(3)]
    return dict(dens=dens, dist=dist, area=area, n_alv=len(F), global_num=global_num, global_area=global_area,
                paso=paso)


def _limpiar(a):
    a.spines[["top", "right"]].set_visible(False)
    a.spines[["left", "bottom"]].set_color("#cfcdc4")
    a.tick_params(colors=GRIS)
    a.grid(axis="y", color="#ecebe6", lw=0.8); a.set_axisbelow(True)


def panel_graficos(res, alv, titulo, ruta=None, dg=None):
    """Área alveolar por réplica · distribución de tamaños · reparto del área por clase."""
    dg = dg or datos_graficos(res, alv)
    reps = dg["dens"]["Réplica"].tolist(); n = len(reps); x = np.arange(n)
    rot = 25 if max(len(r) for r in reps) > 9 else 0
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw=dict(width_ratios=[1, 1.35, 1.1]))

    # (a) densidad alveolar por réplica
    a = ax[0]; d = dg["dens"]["Densidad (alv/cm²)"].values; m = d.mean()
    de = d.std(ddof=1) if n > 1 else 0
    a.bar(x, d, width=0.55, color="#2a78d6", alpha=0.9)
    for i, v in enumerate(d):
        a.text(i, v - max(d.max(), 1e-9) * 0.03, _c(v), ha="center", va="top", color="white", fontsize=11, weight="bold")
    a.axhline(m, color=TINTA, lw=1.4, ls="--")
    a.set_xticks(x); a.set_xticklabels(reps, color=TINTA, rotation=rot, ha="right" if rot else "center")
    a.set_ylim(0, max(d.max(), 1) * 1.25); a.set_ylabel("alvéolos / cm²", color=GRIS)
    a.set_title(f"Densidad alveolar (media {_c(m)} ± {_c(de)})", loc="left", color=TINTA, fontsize=12, pad=44)
    _limpiar(a)

    # (b) distribución de tamaños por diámetro de Feret
    a = ax[1]; dist = dg["dist"]; paso = dg["paso"]
    a.bar(dist["Centro (mm)"], dist["% de alvéolos"], width=paso * 0.9,
          color=["#" + HEX_CLASE[NOMBRES_CLASE.index(c)] for c in dist["Clase"]])
    top = max(dist["% de alvéolos"].max(), 1)
    lims = _limites_clase()
    for xl in lims:
        a.axvline(xl, color=GRIS, lw=1, ls=":")
    x0 = dist["Centro (mm)"].min() - paso / 2; tope = dist["Centro (mm)"].max() + paso / 2
    marcas = [x0] + lims + [tope]
    for (xa, xb), t in zip(zip(marcas[:-1], marcas[1:]), ("finos", "medios", "grandes")):
        a.text((xa + xb) / 2, top * 1.08, t, ha="center", color=TINTA, fontsize=10)
    a.set_ylim(0, top * 1.2); a.set_xlim(x0, tope)
    a.set_xlabel("diámetro de Feret (mm)", color=GRIS); a.set_ylabel("% de alvéolos", color=GRIS)
    a.set_title(f"Distribución de tamaños ({dg['n_alv']} alvéolos, {n} réplicas)", loc="left", color=TINTA,
                fontsize=12, pad=44)
    _limpiar(a)

    # (c) área alveolar por clase y total
    a = ax[2]; ar = dg["area"]; base_ = np.zeros(n)
    cols = [c for c in ar.columns if c.startswith("% área")]
    for k, c in enumerate(cols):
        v = ar[c].values
        a.bar(x, v, bottom=base_, width=0.55, color="#" + HEX_CLASE[k], edgecolor="white", linewidth=1.5,
              label=NOMBRES_CLASE[k])
        for i in range(n):
            if v[i] >= 7:
                a.text(i, base_[i] + v[i] / 2, f"{v[i]:.0f} %", ha="center", va="center", fontsize=9,
                       color="white" if k == 2 else TINTA)
        base_ += v
    for i, fa in enumerate(ar["Área alveolar (% de la superficie)"]):
        a.text(i, 102, f"{_c(fa)} %", ha="center", va="bottom", color=TINTA, fontsize=9.5, weight="bold")
    a.set_xticks(x); a.set_xticklabels(reps, color=TINTA, rotation=rot, ha="right" if rot else "center")
    a.set_ylim(0, 112); a.set_ylabel("% del área alveolar", color=GRIS)
    a.set_title("Área alveolar por clase de tamaño", loc="left", color=TINTA, fontsize=12, pad=44)
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9,
             handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)
    a.text(0, 1.115, "Cifra sobre cada barra: área alveolar (% de la superficie), variable primaria",
           transform=a.transAxes, ha="left", va="bottom", fontsize=8.5, color=GRIS)
    _limpiar(a)

    fig.suptitle(titulo, fontsize=14, weight="bold", x=0.01, ha="left", color=TINTA)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if ruta:
        plt.savefig(ruta, dpi=140, facecolor="white")
    return fig

# ---------- Excel: tablas base + gráficos nativos + figuras ----------
def excel_resultados(ruta, tabla, res, alv_out, dg, meta, figuras):
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, LineChart, Reference
    from openpyxl.chart.label import DataLabelList
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.drawing.image import Image as XLImage

    cab_fill = PatternFill("solid", fgColor=ROJO); cab_font = Font(bold=True, color="FFFFFF")
    resal = PatternFill("solid", fgColor="EEF4FC"); fino = Side(style="thin", color="D9D7CF")
    borde = Border(left=fino, right=fino, top=fino, bottom=fino)

    def tabla_ws(ws, df, fila, col=1, formatos=None, resaltar=()):
        for j, c in enumerate(df.columns):
            cel = ws.cell(row=fila, column=col + j, value=c)
            cel.fill, cel.font, cel.border = cab_fill, cab_font, borde
            cel.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for i, (_, r) in enumerate(df.iterrows(), start=1):
            for j, c in enumerate(df.columns):
                v = r[c]
                v = None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)
                cel = ws.cell(row=fila + i, column=col + j, value=v); cel.border = borde
                if isinstance(v, float):
                    cel.number_format = (formatos or {}).get(c, "0.0" if str(c).startswith(("N ", "n ")) else "0.00")
                if str(r.iloc[0]) in resaltar:
                    cel.fill = resal; cel.font = Font(bold=True)
        return fila + len(df)

    def ancho(ws, anchos):
        for i, w in enumerate(anchos, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w

    def preparar_grafico(ch, titulo, ytit=None, xtit=None):
        ch.title = titulo; ch.style = 10
        if ytit: ch.y_axis.title = ytit
        if xtit: ch.x_axis.title = xtit
        ch.x_axis.delete = False; ch.y_axis.delete = False
        ch.y_axis.majorGridlines = None if False else ch.y_axis.majorGridlines
        ch.height, ch.width = 8.5, 17
        if ch.legend is not None:
            ch.legend.position = "t"
        return ch

    def _solo_valor(DL, fmt):
        dl = DL(); dl.showVal = True; dl.showSerName = False; dl.showCatName = False
        dl.showLegendKey = False; dl.showPercent = False; dl.numFmt = fmt
        return dl

    def imprimir(ws):
        ws.page_setup.orientation = "landscape"; ws.page_setup.paperSize = ws.PAPERSIZE_A4
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.fitToWidth = 1; ws.page_setup.fitToHeight = 0

    wb = Workbook()
    # Hoja 1: resumen
    ws = wb.active; ws.title = "Resumen"
    try:
        logo = XLImage(_logo()); logo.height, logo.width = 48, 127; ws.add_image(logo, "A1")
    except Exception:
        pass
    ws["D1"] = "Ficha de resultados: análisis alveolar de miga"; ws["D1"].font = Font(bold=True, size=14, color=ROJO)
    ws["D2"] = f"{AUTORIA[0]}{AUTORIA[1]} · {DEPARTAMENTO}"; ws["D2"].font = Font(italic=True, size=9, color=SUAVE)
    f = 5
    for k, v in meta["identificacion"]:
        ws.cell(row=f, column=1, value=k).font = Font(bold=True)
        ws.cell(row=f, column=3, value=v); f += 1
    f += 1
    ws.cell(row=f, column=1, value="Tabla 1. Resultados por réplica, media y desviación estándar (DE)").font = \
        Font(bold=True, color=ROJO)
    tabla_ws(ws, tabla, f + 1, resaltar=FILAS_RESUMEN)
    ancho(ws, [16, 18] + [14] * (len(tabla.columns) - 2))
    ws.freeze_panes = "A5"; imprimir(ws)

    # Hoja 2: gráficos con sus tablas base
    wg = wb.create_sheet("Graficos")
    wg["A1"] = "Gráficos y tablas de datos base"; wg["A1"].font = Font(bold=True, size=13, color=ROJO)
    # (a) densidad
    f0 = 3; wg.cell(row=f0, column=1, value="A. Densidad alveolar por réplica").font = Font(bold=True)
    fin = tabla_ws(wg, dg["dens"], f0 + 1)
    n = len(dg["dens"])
    barras = BarChart(); barras.type = "col"
    barras.add_data(Reference(wg, min_col=2, min_row=f0 + 1, max_row=fin), titles_from_data=True)
    barras.set_categories(Reference(wg, min_col=1, min_row=f0 + 2, max_row=fin))
    barras.series[0].graphicalProperties.solidFill = "2A78D6"
    barras.series[0].graphicalProperties.line.solidFill = "2A78D6"
    barras.dataLabels = _solo_valor(DataLabelList, "0.0")
    barras.y_axis.scaling.min = 0
    linea = LineChart()
    linea.add_data(Reference(wg, min_col=3, min_row=f0 + 1, max_row=fin), titles_from_data=True)
    linea.series[0].graphicalProperties.line.solidFill = TINTA_D
    linea.series[0].graphicalProperties.line.dashStyle = "dash"
    linea.series[0].graphicalProperties.line.width = 19050
    linea.series[0].marker.symbol = "none"
    barras.gapWidth = 80
    barras += linea
    wg.add_chart(preparar_grafico(barras, "Densidad alveolar por réplica", "alvéolos/cm²"), "F3")
    # (b) distribución
    f1 = max(fin + 3, f0 + 20)
    wg.cell(row=f1, column=1, value="B. Distribución de tamaños (todas las réplicas)").font = Font(bold=True)
    fin = tabla_ws(wg, dg["dist"], f1 + 1)
    dch = BarChart(); dch.type = "col"; dch.grouping = "stacked"; dch.overlap = 100; dch.gapWidth = 15
    dch.y_axis.scaling.min = 0
    dch.add_data(Reference(wg, min_col=5, max_col=7, min_row=f1 + 1, max_row=fin), titles_from_data=True)
    dch.set_categories(Reference(wg, min_col=1, min_row=f1 + 2, max_row=fin))
    for s, hx in zip(dch.series, HEX_CLASE):
        s.graphicalProperties.solidFill = hx; s.graphicalProperties.line.solidFill = hx
    wg.add_chart(preparar_grafico(dch, "Distribución de tamaños por diámetro de Feret", "% de alvéolos",
                                  "diámetro de Feret (mm)"), f"J{f1}")
    # (c) área por clase
    f2 = max(fin + 3, f1 + 20)
    wg.cell(row=f2, column=1, value="C. Reparto del área alveolar por clase").font = Font(bold=True)
    fin = tabla_ws(wg, dg["area"], f2 + 1)
    ach = BarChart(); ach.type = "col"; ach.grouping = "percentStacked"; ach.overlap = 100; ach.gapWidth = 60
    ach.y_axis.scaling.min = 0
    ach.add_data(Reference(wg, min_col=2, max_col=4, min_row=f2 + 1, max_row=fin), titles_from_data=True)
    ach.set_categories(Reference(wg, min_col=1, min_row=f2 + 2, max_row=fin))
    for s, hx in zip(ach.series, HEX_CLASE):
        s.graphicalProperties.solidFill = hx; s.graphicalProperties.line.solidFill = "FFFFFF"
    ach.dataLabels = _solo_valor(DataLabelList, "0")
    wg.add_chart(preparar_grafico(ach, "Área alveolar por clase de tamaño", "% del área alveolar"), f"G{f2}")
    ancho(wg, [24, 18, 16, 16, 16, 16, 16]); imprimir(wg)

    # Hoja 3: alvéolos
    wa = wb.create_sheet("Alveolos")
    tabla_ws(wa, alv_out, 1, formatos={c: "0.000" for c in alv_out.columns})
    ancho(wa, [16, 12, 18, 12, 16, 16, 30]); wa.freeze_panes = "A2"; imprimir(wa)

    # Hoja 4: figuras
    wf = wb.create_sheet("Figuras"); fila = 1
    for tit, p, ancho_px in figuras:
        wf.cell(row=fila, column=1, value=tit).font = Font(bold=True, color=ROJO)
        try:
            im = XLImage(p); r_ = ancho_px / im.width; im.width, im.height = ancho_px, int(im.height * r_)
            wf.add_image(im, f"A{fila + 1}"); fila += int(im.height / 20) + 4
        except Exception:
            fila += 2
    # Hoja 2: validación (blanco control y réplicas)
    if meta.get("validacion") is not None:
        wv = wb.create_sheet("Validacion", 1)
        wv["A1"] = "Validación del análisis: blanco control y réplicas"
        wv["A1"].font = Font(bold=True, size=13, color=ROJO)
        fin_v = tabla_ws(wv, meta["validacion"], 3)
        rel = {"OK": "E9F6EE", "REVISAR": "FFF6E5", "PENDIENTE": "EEF4FC", "NO VÁLIDO": "FDECEC"}
        col_e = list(meta["validacion"].columns).index("Estado") + 1
        for fila in range(4, fin_v + 1):
            cel = wv.cell(row=fila, column=col_e)
            if cel.value in rel:
                cel.fill = PatternFill("solid", fgColor=rel[cel.value]); cel.font = Font(bold=True)
        for fila in range(3, fin_v + 1):
            for col in range(1, len(meta["validacion"].columns) + 1):
                wv.cell(row=fila, column=col).alignment = Alignment(wrap_text=True, vertical="top")
        wv.cell(row=fin_v + 2, column=1, value="Control no automatizable: mirar el detalle ampliado de la figura de "
                "segmentación (hoja Figuras). Los contornos deben caer sobre alvéolos, no sobre pared.").font = \
            Font(italic=True, color=SUAVE)
        ancho(wv, [34, 30, 30, 14, 70]); imprimir(wv)
    wb.save(ruta)




# ---------- Contenido de la ficha técnica ----------
def contenido_ficha(muestra, tabla, res, dg, opciones, avisos, figs, fecha):
    t = tabla.set_index("Réplica"); reps = res.emplazamiento.tolist(); n = len(reps)
    fila = "Media" if "Media" in t.index else reps[0]
    de = t.loc["DE"] if "DE" in t.index else None
    g = lambda c: t.loc[fila, c]
    ds = lambda c: (f" ± {_c(de[c], 2 if 'mm²' in c else 1)}" if de is not None else "")
    codigo = f"ALV-{fecha:%Y%m%d-%H%M}"
    dens = res.densidad_alv_cm2
    cv = 100 * dens.std() / dens.mean() if n > 1 and dens.mean() else 0
    imin, imax = dens.idxmin(), dens.idxmax()
    gn = [res[f"% {c}"].mean() for c in CLASES_NOMBRES_RES]
    ga = [dg["area"][c].mean() for c in [c for c in dg["area"].columns if c.startswith("% área")]]
    orden = sorted(range(3), key=lambda k: -gn[k]); nomb = ["finos", "medios", "grandes"]
    escala_txt = ("automática, a partir de la regla de cada fotografía" if opciones["escala"] is None
                  else f"indicada por el analista ({_c(opciones['escala'], 2)} px/mm)")
    umbral_txt = {"otsu": "Otsu", "li": "Li", "media": "media"}[opciones["umbral"]]

    ident = [("Código del informe", codigo), ("Muestra", muestra),
             ("Réplicas analizadas", f"{n} ({'; '.join(reps)})"),
             ("Archivos de imagen", "; ".join(res.archivo.tolist())),
             ("Fecha de análisis", f"{fecha:%d/%m/%Y %H:%M}"),
             ("Método", f"Análisis de imagen en escala de grises: umbral de {umbral_txt} y separación por watershed"),
             ("Zona analizada", f"Cuadrado central de {_c(opciones['lado'], 0)} × {_c(opciones['lado'], 0)} mm por réplica"),
             ("Escala", escala_txt[0].upper() + escala_txt[1:]),
             ("Alvéolo mínimo contado", f"Diámetro equivalente ≥ {_c(opciones['dmin'], 1)} mm")]
    kpi = [(_c(g("Densidad (alv/cm²)")), "Densidad alveolar", f"alv/cm²{ds('Densidad (alv/cm²)')}"),
           (f"{g('N equiv. 3×3 cm'):.0f}", "N equivalente 3 × 3 cm", "alvéolos"),
           (" / ".join(f"{g(c):.0f}" for c in _cols_pct()),
            "Finos / medios / grandes", "% del número"),
           (f"{_c(g('Área alveolar (%)'))} %", "Área alveolar", f"de la superficie{ds('Área alveolar (%)')}"),
           (_c(g("Área media (mm²)"), 2), "Área media por alvéolo", f"mm²{ds('Área media (mm²)')}")]
    cols_t1 = (["Réplica", "Densidad (alv/cm²)", "N equiv. 3×3 cm"] + _cols_pct()
               + ["Área alveolar (%)", "Área media (mm²)", "Feret mediano (mm)"])
    cab_t1 = ["Réplica", "Densidad (alv/cm²)", "N equiv. 3 × 3 cm", "Finos (%)", "Medios (%)", "Grandes (%)",
              "Área alveolar (%)", "Área media (mm²)", "Feret mediano (mm)"]
    filas_t1 = []
    for _, r in tabla[cols_t1].iterrows():
        filas_t1.append([str(r["Réplica"])] + [_c(r[c], 0 if c.startswith("N ") else
                                                  (1 if c.startswith(("%", "Densidad", "Área alveolar")) else 2))
                                               for c in cols_t1[1:]])
    descr = [
        f"La densidad alveolar media fue de **{_c(g('Densidad (alv/cm²)'))}{ds('Densidad (alv/cm²)')} alv/cm²**"
        + (f" (coeficiente de variación entre réplicas: {_c(cv)} %)" if n > 1 else "")
        + f", equivalente a {g('N equiv. 3×3 cm'):.0f} alvéolos en la plantilla de 3 × 3 cm.",
        (f"La densidad varió entre {_c(dens[imin])} alv/cm² ({reps[imin]}) y {_c(dens[imax])} alv/cm² ({reps[imax]})."
         if n > 1 else ""),
        f"Por número (media de las réplicas), predominaron los alvéolos {nomb[orden[0]]} ({gn[orden[0]]:.0f} %), seguidos de los "
        f"{nomb[orden[1]]} ({gn[orden[1]]:.0f} %) y los {nomb[orden[2]]} ({gn[orden[2]]:.0f} %).",
        f"Por superficie (media de las réplicas), los alvéolos finos aportaron el {ga[0]:.0f} % del área alveolar, los medios el "
        f"{ga[1]:.0f} % y los grandes el {ga[2]:.0f} %.",
        f"La fracción de área alveolar fue de {_c(g('Área alveolar (%)'))}{ds('Área alveolar (%)')} % y el área media "
        f"por alvéolo, de {_c(g('Área media (mm²)'), 2)}{ds('Área media (mm²)')} mm².",
        f"El diámetro de Feret mediano fue de {_c(g('Feret mediano (mm)'), 2)} mm y el percentil 90, de "
        f"{_c(g('Feret P90 (mm)'), 2)} mm.",
    ]
    descr = [d for d in descr if d]
    qc_cab = ["Réplica", "Archivo", "Escala (px/mm)", "Origen de la escala", "Zona dentro de la ventana (%)"]
    org = {"regla": "Regla (automática)", "indicada": "Indicada por el analista"}
    qc = [[r.emplazamiento, r.archivo, _c(r.escala_px_mm, 2), org.get(r.origen_escala, r.origen_escala),
           f"{r.dentro_pct:.0f}"]
          for r in res.itertuples()]
    av = [re.sub(r"^[⚠✔]\s*", "", a) for a in avisos if "⚠" in a]

    B = []
    B.append(("titulo", "FICHA TÉCNICA DE ANÁLISIS ALVEOLAR DE MIGA",
              f"Práctica {PRACTICA} · Panificación y Pastelería · Muestra {muestra}"))
    B.append(("kv", ident))
    B.append(("h", "Resumen de resultados"))
    B.append(("kpi", kpi))
    B.append(("h", "1. Objetivo"))
    B.append(("p", f"Caracterizar la estructura alveolar de la miga de la muestra **{muestra}** mediante análisis de "
                   f"imagen en escala de grises a partir de {n} réplicas fotográficas, y expresar el resultado como "
                   f"densidad alveolar, distribución de tamaños y fracción de área alveolar."))
    B.append(("h", "2. Material y método"))
    B.append(("p", "Cada réplica es una fotografía de la superficie de corte de la miga sobre una plantilla con "
                   "ventana de 3 × 3 cm y una regla milimetrada en el mismo plano. Todas las réplicas se procesaron "
                   "con el mismo flujo y los mismos parámetros."))
    B.append(("bullets", [
        "**Calibración.** La escala (px/mm) se obtiene de la periodicidad de las marcas de la regla en escala de "
        "grises y se contrasta con el tamaño de la ventana de 3 cm.",
        f"**Zona de análisis.** Cuadrado central de {_c(opciones['lado'], 0)} × {_c(opciones['lado'], 0)} mm dentro de "
        f"la ventana, para excluir los bordes y la sombra del corte.",
        f"**Preprocesado.** Conversión a 8 bits y sustracción del fondo (radio de {_c(FONDO_MM)} mm) para corregir la "
        "iluminación desigual.",
        f"**Segmentación.** Umbral automático de {umbral_txt}, con un contraste mínimo de {CONTRASTE_MIN:g} niveles de gris: los píxeles oscuros se asignan a alvéolo y los claros "
        f"a pared de miga. Se aplican apertura morfológica, relleno de huecos y separación por watershed.",
        "**Medición y clasificación.** Área, diámetro equivalente, diámetro de Feret y relación de ejes de cada "
        f"alvéolo. Clases por diámetro de Feret: {'; '.join(NOMBRES_CLASE)}."]))
    B.append(("tabla", ["Variable", "Cálculo", "Unidad"],
              [["Densidad alveolar", "N / área analizada", "alv/cm²"],
               ["N equivalente 3 × 3 cm", "9 × densidad alveolar", "alvéolos"],
               ["Porcentaje por clase", "n de la clase / N × 100", "%"],
               ["Fracción de área alveolar", "Σ área de alvéolos / área analizada × 100", "%"],
               ["Diámetro equivalente", "2 × √(área / π)", "mm"]],
              [0.32, 0.48, 0.20], "Tabla 1. Variables calculadas.", ()))
    B.append(("p", f"N se cuenta con marco insesgado: no se incluyen los alvéolos que tocan el borde derecho o el "
                   f"inferior de la zona analizada. La fracción de área alveolar sí los incluye. Se cuentan los "
                   f"alvéolos con diámetro equivalente igual o superior a {_c(opciones['dmin'], 1)} mm. Los resultados "
                   f"se expresan como media ± desviación estándar (DE) de las réplicas."))
    B.append(("h", "3. Resultados"))
    B.append(("tabla", cab_t1, filas_t1, [0.14, 0.11, 0.10, 0.085, 0.09, 0.10, 0.11, 0.12, 0.145],
              "Tabla 2. Resultados por réplica, media, desviación estándar (DE), CV y recorrido.", FILAS_RESUMEN))
    B.append(("bullets", descr))
    B.append(("img", figs["segmentacion"], 0.80,
              "Figura 1. Zona analizada en color, imagen en 8 bits con sustracción de fondo y alvéolos "
              "segmentados y clasificados por diámetro de Feret (azul: finos; naranja: medios; rojo: grandes)."))
    B.append(("img", figs["graficos"], 1.0,
              "Figura 2. Densidad alveolar por réplica (línea discontinua: media), distribución de tamaños por "
              "diámetro de Feret del conjunto de réplicas y reparto del área alveolar por clase de tamaño."))
    B.append(("h", "4. Control de calidad"))
    B.append(("tabla", qc_cab, qc, [0.25, 0.22, 0.13, 0.20, 0.20], "Tabla 3. Calibración y zona analizada por réplica.", ()))
    B.append(("p", "Avisos registrados durante el análisis: " + ("; ".join(av) + "." if av else "ninguno.")))
    B.append(("h", "5. Alcance y limitaciones"))
    B.append(("bullets", [
        "Los resultados describen la superficie de corte fotografiada. Si las réplicas proceden de la misma pieza, "
        "son submuestras y no repeticiones independientes del proceso.",
        "La fracción de área alveolar depende del método de umbral: solo es comparable entre muestras analizadas "
        "con el mismo método.",
        f"El análisis de imagen detecta poros desde {_c(opciones['dmin'], 2)} mm que no se aprecian a simple vista, por lo que su recuento "
        "no es equivalente al recuento manual.",
        "El enfoque, la perpendicularidad de la cámara, la iluminación y la limpieza del corte influyen en el "
        "resultado."]))
    B.append(("h", "6. Observaciones y validación"))
    B.append(("firma", None))
    pie = f"Ficha técnica {codigo} · Análisis alveolar {PRACTICA} · Generada el {fecha:%d/%m/%Y}"
    return B, pie, codigo




def _trozos(texto):
    """Divide un texto con **negrita** en [(texto, negrita)]."""
    partes = re.split(r"(\*\*[^*]+\*\*)", texto)
    return [(p[2:-2], True) if p.startswith("**") else (p, False) for p in partes if p]


# ---------- Ficha técnica en Word (editable) ----------
def ficha_docx(ruta, B, pie):
    if not _asegurar("python-docx", "docx"):
        raise RuntimeError("no se pudo instalar python-docx")
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor, Emu
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    rgb = lambda h: RGBColor.from_string(h)
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10); st.font.color.rgb = rgb(TINTA_D)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    st.paragraph_format.space_after = Pt(4); st.paragraph_format.line_spacing = 1.1
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.6); sec.top_margin = Cm(2.6); sec.bottom_margin = Cm(1.8)
    sec.header_distance = Cm(0.8); sec.footer_distance = Cm(0.8)
    util = sec.page_width - sec.left_margin - sec.right_margin

    def sombra(celda, hexcol):
        tcPr = celda._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcol); tcPr.append(shd)

    def sin_bordes(tabla):
        tblPr = tabla._tbl.tblPr; b = OxmlElement("w:tblBorders")
        for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{lado}"); e.set(qn("w:val"), "nil"); b.append(e)
        tblPr.append(b)

    def bordes_finos(tabla, color="D9D7CF"):
        tblPr = tabla._tbl.tblPr; b = OxmlElement("w:tblBorders")
        for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{lado}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4")
            e.set(qn("w:color"), color); b.append(e)
        tblPr.append(b)

    def linea_inferior(p, color=ROJO, sz="8"):
        pPr = p._p.get_or_add_pPr(); bd = OxmlElement("w:pBdr"); bt = OxmlElement("w:bottom")
        bt.set(qn("w:val"), "single"); bt.set(qn("w:sz"), sz); bt.set(qn("w:space"), "2"); bt.set(qn("w:color"), color)
        bd.append(bt); pPr.append(bd)

    def escribir(p, texto, size=None, color=None, italic=False, bold=None):
        for t, b in _trozos(texto):
            r = p.add_run(t); r.bold = bold if bold is not None else b; r.italic = italic
            if size: r.font.size = Pt(size)
            if color: r.font.color.rgb = rgb(color)
        return p

    def campo(p, codigo):
        r = p.add_run(); f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = codigo
        f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
        r._r.append(f1); r._r.append(it); r._r.append(f2); r.font.size = Pt(7.5); r.font.color.rgb = rgb(SUAVE)

    # Encabezado como en los documentos de trabajo: logo a la izquierda, autoría a la derecha
    hdr = sec.header; hdr.is_linked_to_previous = False
    ht = hdr.add_table(rows=1, cols=2, width=util); sin_bordes(ht)
    ht.columns[0].width = ht.cell(0, 0).width = Cm(7); ht.cell(0, 1).width = util - Cm(7)
    ht.cell(0, 0).paragraphs[0].add_run().add_picture(_logo(), width=Cm(4.2))
    p = ht.cell(0, 1).paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run(AUTORIA[0]); r.bold = True; r.font.color.rgb = rgb(ROJO); r.font.size = Pt(8.5)
    r = p.add_run(AUTORIA[1]); r.bold = True; r.italic = True; r.font.size = Pt(8.5)
    p2 = ht.cell(0, 1).add_paragraph(); p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p2.add_run(DEPARTAMENTO); r.italic = True; r.font.size = Pt(8.5)
    if hdr.paragraphs and not hdr.paragraphs[0].text:
        hdr.paragraphs[0]._p.getparent().remove(hdr.paragraphs[0]._p)
    fp = sec.footer.paragraphs[0]
    r = fp.add_run(pie + " · Página "); r.font.size = Pt(7.5); r.font.color.rgb = rgb(SUAVE)
    campo(fp, "PAGE"); r = fp.add_run(" de "); r.font.size = Pt(7.5); r.font.color.rgb = rgb(SUAVE)
    campo(fp, "NUMPAGES")

    for b in B:
        tipo = b[0]
        if tipo == "titulo":
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            escribir(p, b[1], 15, ROJO, bold=True); p.paragraph_format.space_after = Pt(2)
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            escribir(p, b[2], 9.5, SUAVE, italic=True); p.paragraph_format.space_after = Pt(10)
        elif tipo == "h":
            p = doc.add_paragraph(); escribir(p, b[1].upper(), 10.5, ROJO, bold=True)
            p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.keep_with_next = True; linea_inferior(p)
        elif tipo == "p":
            p = doc.add_paragraph(); escribir(p, b[1]); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif tipo == "bullets":
            for t in b[1]:
                p = doc.add_paragraph(style="List Bullet"); escribir(p, t); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.space_after = Pt(2)
        elif tipo == "kv":
            t = doc.add_table(rows=len(b[1]), cols=2); bordes_finos(t); t.alignment = WD_TABLE_ALIGNMENT.CENTER
            t.autofit = False; t.columns[0].width = Cm(5); t.columns[1].width = util - Cm(5)
            for i, (k, v) in enumerate(b[1]):
                c0, c1 = t.cell(i, 0), t.cell(i, 1); c0.width, c1.width = Cm(5), util - Cm(5)
                sombra(c0, "F6F5F1"); escribir(c0.paragraphs[0], k, 9, TINTA_D, bold=True)
                escribir(c1.paragraphs[0], str(v), 9)
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
        elif tipo == "kpi":
            t = doc.add_table(rows=1, cols=len(b[1])); bordes_finos(t, "FFFFFF")
            for j, (val, et, det) in enumerate(b[1]):
                c = t.cell(0, j); sombra(c, "F3F7FC"); c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER; escribir(p, et, 8, SUAVE)
                p = c.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; escribir(p, val, 15, AZUL_D, bold=True)
                p.paragraph_format.space_after = Pt(0)
                p = c.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; escribir(p, det, 8, SUAVE)
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
        elif tipo == "tabla":
            _, cab, filas, pesos, titulo, resaltar = b
            p = doc.add_paragraph(); escribir(p, titulo, 9, TINTA_D, bold=True); p.paragraph_format.keep_with_next = True
            t = doc.add_table(rows=1 + len(filas), cols=len(cab)); bordes_finos(t); t.alignment = WD_TABLE_ALIGNMENT.CENTER
            t.autofit = False
            for j in range(len(cab)):
                t.columns[j].width = Emu(int(util * pesos[j]))
            for j, h in enumerate(cab):
                c = t.cell(0, j); sombra(c, ROJO); c.width = Emu(int(util * pesos[j]))
                pp = c.paragraphs[0]; pp.alignment = WD_ALIGN_PARAGRAPH.CENTER; escribir(pp, h, 8, "FFFFFF", bold=True)
            for i, fila in enumerate(filas, start=1):
                for j, v in enumerate(fila):
                    c = t.cell(i, j); c.width = Emu(int(util * pesos[j]))
                    pp = c.paragraphs[0]; pp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    destacada = str(fila[0]) in resaltar
                    if destacada:
                        sombra(c, "EEF4FC")
                    escribir(pp, str(v), 8.5, TINTA_D, bold=destacada)
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
        elif tipo == "img":
            _, ruta_img, frac, pie_f = b
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(ruta_img, width=Emu(int(util * frac)))
            p.paragraph_format.keep_with_next = True
            p = doc.add_paragraph(); escribir(p, pie_f, 8.5, SUAVE, italic=True); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif tipo == "firma":
            p = doc.add_paragraph(); escribir(p, "Observaciones del analista:", 9.5, TINTA_D, bold=True)
            for _ in range(3):
                p = doc.add_paragraph(); linea_inferior(p, "BFBFBF", "4"); p.paragraph_format.space_after = Pt(10)
            t = doc.add_table(rows=2, cols=3); sin_bordes(t)
            for j, et in enumerate(["Analista", "Fecha", "Firma"]):
                p = t.cell(0, j).paragraphs[0]; linea_inferior(p, "7F7F7F", "4"); p.paragraph_format.space_before = Pt(18)
                escribir(t.cell(1, j).paragraphs[0], et, 8.5, SUAVE)
    doc.save(ruta)


# ---------- Ficha técnica en PDF ----------
def ficha_pdf(ruta, B, pie):
    if not _asegurar("reportlab", "reportlab"):
        raise RuntimeError("no se pudo instalar reportlab")
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
    from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
                                    Image as RLImage, KeepTogether, ListFlowable, ListItem)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader

    fdir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    try:
        pdfmetrics.registerFont(TTFont("Sans", os.path.join(fdir, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("Sans-B", os.path.join(fdir, "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("Sans-I", os.path.join(fdir, "DejaVuSans-Oblique.ttf")))
        pdfmetrics.registerFont(TTFont("Sans-BI", os.path.join(fdir, "DejaVuSans-BoldOblique.ttf")))
        from reportlab.lib.fonts import addMapping
        addMapping("Sans", 0, 0, "Sans"); addMapping("Sans", 1, 0, "Sans-B")
        addMapping("Sans", 0, 1, "Sans-I"); addMapping("Sans", 1, 1, "Sans-BI")
        F, FB, FI = "Sans", "Sans-B", "Sans-I"
    except Exception:
        F, FB, FI = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"
    C = lambda h: colors.HexColor("#" + h)
    sty = {
        "p": ParagraphStyle("p", fontName=F, fontSize=8.8, leading=12.2, textColor=C(TINTA_D), alignment=TA_JUSTIFY,
                            spaceAfter=4),
        "h": ParagraphStyle("h", fontName=FB, fontSize=9.6, leading=12, textColor=C(ROJO), spaceBefore=9,
                            spaceAfter=5),
        "tit": ParagraphStyle("tit", fontName=FB, fontSize=14, leading=17, textColor=C(ROJO), alignment=TA_CENTER),
        "sub": ParagraphStyle("sub", fontName=FI, fontSize=8.8, leading=11, textColor=C(SUAVE), alignment=TA_CENTER,
                              spaceAfter=8),
        "cap": ParagraphStyle("cap", fontName=FI, fontSize=7.8, leading=10, textColor=C(SUAVE), alignment=TA_JUSTIFY,
                              spaceBefore=3, spaceAfter=8),
        "tt": ParagraphStyle("tt", fontName=FB, fontSize=8.2, leading=10, textColor=C(TINTA_D), spaceAfter=3),
        "c": ParagraphStyle("c", fontName=F, fontSize=7.6, leading=9.4, textColor=C(TINTA_D), alignment=TA_CENTER),
        "cl": ParagraphStyle("cl", fontName=F, fontSize=7.6, leading=9.4, textColor=C(TINTA_D)),
        "ch": ParagraphStyle("ch", fontName=FB, fontSize=7.4, leading=9, textColor=colors.white, alignment=TA_CENTER),
    }
    md = lambda t: "".join(f"<b>{x}</b>" if b else x for x, b in _trozos(
        t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")))
    W_, H_ = A4; mx = 1.6 * cm; util = W_ - 2 * mx
    logo = ImageReader(_logo()); lw, lh = logo.getSize()

    def pagina(cv, doc_):
        cv.saveState()
        alto = 1.55 * cm; cv.drawImage(logo, mx, H_ - 0.8 * cm - alto, width=alto * lw / lh, height=alto, mask="auto")
        cv.setFont(FB, 8.2); x = W_ - mx
        cv.setFillColor(C(TINTA_D)); w2 = cv.stringWidth(AUTORIA[1], FB, 8.2)
        cv.drawRightString(x, H_ - 1.15 * cm, AUTORIA[1])
        cv.setFillColor(C(ROJO)); cv.drawRightString(x - w2, H_ - 1.15 * cm, AUTORIA[0])
        cv.setFont(FI, 8.2); cv.setFillColor(C(TINTA_D)); cv.drawRightString(x, H_ - 1.55 * cm, DEPARTAMENTO)
        cv.setFont(F, 7); cv.setFillColor(C(SUAVE))
        cv.drawString(mx, 1.0 * cm, pie); cv.drawRightString(W_ - mx, 1.0 * cm, f"Página {doc_.page}")
        cv.setStrokeColor(C("D9D7CF")); cv.setLineWidth(0.5); cv.line(mx, 1.3 * cm, W_ - mx, 1.3 * cm)
        cv.restoreState()

    def encabezado_seccion(texto):
        t = Table([[Paragraph(md(texto.upper()), sty["h"])]], colWidths=[util])
        t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.9, C(ROJO)), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 1), ("TOPPADDING", (0, 0), (-1, -1), 4)]))
        return t

    flujo = []; pendiente_h = None
    for b in B:
        tipo = b[0]; bloque = []
        if tipo == "titulo":
            bloque = [Paragraph(md(b[1]), sty["tit"]), Spacer(1, 3), Paragraph(md(b[2]), sty["sub"])]
        elif tipo == "h":
            pendiente_h = encabezado_seccion(b[1]); continue
        elif tipo == "p":
            bloque = [Paragraph(md(b[1]), sty["p"])]
        elif tipo == "bullets":
            bloque = [ListFlowable([ListItem(Paragraph(md(t), sty["p"]), leftIndent=10, value="•") for t in b[1]],
                                   bulletType="bullet", start="•", leftIndent=10, bulletFontSize=7,
                                   bulletColor=C(ROJO))]
        elif tipo == "kv":
            d = [[Paragraph(md(f"**{k}**"), sty["cl"]), Paragraph(md(str(v)), sty["cl"])] for k, v in b[1]]
            t = Table(d, colWidths=[4.6 * cm, util - 4.6 * cm])
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, C("D9D7CF")),
                                   ("BACKGROUND", (0, 0), (0, -1), C("F6F5F1")),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
            bloque = [t, Spacer(1, 6)]
        elif tipo == "kpi":
            celdas = [[Paragraph(f"<font size=7 color='#{SUAVE}'>{md(et)}</font><br/>"
                                 f"<font name='{FB}' size=14 color='#{AZUL_D}'>{md(val)}</font><br/>"
                                 f"<font size=7 color='#{SUAVE}'>{md(det)}</font>",
                                 ParagraphStyle("k", fontName=F, alignment=TA_CENTER, leading=15))
                       for val, et, det in b[1]]]
            t = Table(celdas, colWidths=[util / len(b[1])] * len(b[1]))
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C("F3F7FC")),
                                   ("LINEAFTER", (0, 0), (-2, -1), 2, colors.white),
                                   ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            bloque = [t, Spacer(1, 4)]
        elif tipo == "tabla":
            _, cab, filas, pesos, titulo, resaltar = b
            d = [[Paragraph(md(h), sty["ch"]) for h in cab]] + \
                [[Paragraph(md(str(v)), sty["cl"] if j == 0 else sty["c"]) for j, v in enumerate(f)] for f in filas]
            t = Table(d, colWidths=[util * p for p in pesos], repeatRows=1)
            est = [("BACKGROUND", (0, 0), (-1, 0), C(ROJO)), ("GRID", (0, 0), (-1, -1), 0.4, C("D9D7CF")),
                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                   ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
            for i, f in enumerate(filas, start=1):
                if str(f[0]) in resaltar:
                    est.append(("BACKGROUND", (0, i), (-1, i), C("EEF4FC")))
                elif i % 2 == 0:
                    est.append(("BACKGROUND", (0, i), (-1, i), C("FAFAF8")))
            t.setStyle(TableStyle(est))
            bloque = [Paragraph(md(titulo), sty["tt"]), t, Spacer(1, 7)]
        elif tipo == "img":
            _, ruta_img, frac, pie_f = b
            ir = ImageReader(ruta_img); iw, ih = ir.getSize(); w = util * frac; h = w * ih / iw
            hmax = H_ - 6.5 * cm
            if h > hmax:
                h = hmax; w = h * iw / ih
            bloque = [KeepTogether([RLImage(ruta_img, width=w, height=h), Paragraph(md(pie_f), sty["cap"])])]
        elif tipo == "firma":
            lineas = Table([[""]] * 3, colWidths=[util], rowHeights=[0.75 * cm] * 3)
            lineas.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, C("BFBFBF"))]))
            firmas = Table([["", "", ""], ["Analista", "Fecha", "Firma"]], colWidths=[util / 3] * 3,
                           rowHeights=[1.1 * cm, 0.5 * cm])
            firmas.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.5, C("7F7F7F")),
                                        ("FONT", (0, 1), (-1, 1), F, 7.6), ("TEXTCOLOR", (0, 1), (-1, 1), C(SUAVE)),
                                        ("RIGHTPADDING", (0, 0), (-1, -1), 14)]))
            bloque = [Paragraph(md("**Observaciones del analista:**"), sty["p"]), lineas, Spacer(1, 6), firmas]
        if pendiente_h is not None:
            flujo.append(KeepTogether([pendiente_h] + bloque[:2])); flujo += bloque[2:]; pendiente_h = None
        else:
            flujo += bloque
    doc_ = BaseDocTemplate(ruta, pagesize=A4, leftMargin=mx, rightMargin=mx, topMargin=2.7 * cm,
                           bottomMargin=1.8 * cm, title="Ficha técnica de análisis alveolar", author=AUTORIA[1])
    doc_.addPageTemplates([PageTemplate(id="p", frames=[Frame(mx, 1.8 * cm, util, H_ - 4.5 * cm, id="f", leftPadding=0, rightPadding=0,
                                                                     topPadding=0, bottomPadding=0)],
                                        onPage=pagina)])
    doc_.build(flujo)



def _cols_pct():
    return [COLUMNAS[f"% {nom}"] for nom in NOMBRES_CLASE]


def generar_informes(base, muestra, tabla, res, alv, alv_out, dg, opciones, avisos, val=None):
    fecha = datetime.datetime.now()
    figs = {"segmentacion": base + "_figura_segmentacion.png", "graficos": base + "_graficos.png"}
    B, pie, codigo = contenido_ficha(muestra, tabla, res, dg, opciones, avisos, figs, fecha)
    meta = {"identificacion": B[1][1], "validacion": val}
    hechos, fallos = [], []
    tareas = [("Excel", lambda: excel_resultados(base + "_resultados.xlsx", tabla, res, alv_out, dg, meta,
                                                          [("Figura 1. Segmentación por réplica", figs["segmentacion"], 1100),
                                                           ("Figura 2. Gráficos de resultados", figs["graficos"], 1100)]))]
    if FICHA_TECNICA:
        tareas += [("Word", lambda: ficha_docx(base + "_ficha_tecnica.docx", B, pie)),
                   ("PDF", lambda: ficha_pdf(base + "_ficha_tecnica.pdf", B, pie))]
    for nombre, fn in tareas:
        try:
            fn(); hechos.append(nombre)
        except Exception as e:
            fallos.append(f"{nombre} ({e})")
    return hechos, fallos




# ============================================================
#  REPORTE PARA LA HOJA D · MIGA (Word, formato UBU)
# ============================================================
COLS_HOJA_D = [("escala_foto_px_mm", "Escala (px/mm)", 2), ("area_alveolar_%", "Área alveolar (%)", 1),
               ("densidad_alv_cm2", "Densidad (alv/cm²)", 1), ("feret_p90_mm", "Feret P90 (mm)", 2)]


def _cols_hoja_d():
    nc = _nombres_clase()
    return COLS_HOJA_D + [(f"% {nc[0]}", "% finos < 0,5 mm", 1), (f"% {nc[1]}", "% medios 0,5–1 mm", 1),
                          ("%_area_gt2mm", f"% del área en alvéolos > {_es(LIM_MUY_GRANDES_MM, 0)} mm", 1)]


def reporte_hoja_d(ruta, reg, blanco, grupos=None, comparar=True):
    """Reporte de las salidas del aplicativo ordenado como la Hoja D · Miga: D.1 blanco, D.2 réplicas con media y
    recorrido, D.3 validación por grupo y D.4 lectura de cada par. Solo transcribe datos: no interpreta."""
    if not _asegurar("python-docx", "docx"):
        raise RuntimeError("no se pudo instalar python-docx")
    import datetime as _dt
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor, Emu
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    reg = reg.copy()
    todos = list(dict.fromkeys(reg.muestra)) if len(reg) else []
    grupos = [g for g in (grupos or todos) if g in todos]
    rgb = lambda h: RGBColor.from_string(h)
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10); st.font.color.rgb = rgb(TINTA_D)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    st.paragraph_format.space_after = Pt(3); st.paragraph_format.line_spacing = 1.08
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.5); sec.top_margin = Cm(2.5); sec.bottom_margin = Cm(1.5)
    sec.header_distance = Cm(0.7); sec.footer_distance = Cm(0.6)
    util = sec.page_width - sec.left_margin - sec.right_margin

    def _antes_de(padre, elem, sucesores):
        """Inserta respetando el orden del esquema OOXML (Word puede dar el archivo por dañado si no)."""
        for hijo in padre:
            if hijo.tag in [qn(x) for x in sucesores]:
                hijo.addprevious(elem); return
        padre.append(elem)

    def sombra(c, hexcol):
        tcPr = c._tc.get_or_add_tcPr()
        for viejo in tcPr.findall(qn("w:shd")):
            tcPr.remove(viejo)
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcol)
        _antes_de(tcPr, shd, ("w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign", "w:hideMark"))

    def bordes(t, val="single", color="B8B8B8"):
        b = OxmlElement("w:tblBorders")
        for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{lado}"); e.set(qn("w:val"), val)
            if val != "nil":
                e.set(qn("w:sz"), "4"); e.set(qn("w:color"), color)
            b.append(e)
        _antes_de(t._tbl.tblPr, b, ("w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription"))

    def txt(p, texto, size=None, color=None, bold=False, italic=False):
        r = p.add_run(texto); r.bold = bold; r.italic = italic
        if size: r.font.size = Pt(size)
        if color: r.font.color.rgb = rgb(color)
        return r

    def campo(p, codigo):
        r = p.add_run(); f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = codigo
        f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
        r._r.append(f1); r._r.append(it); r._r.append(f2); r.font.size = Pt(7.5); r.font.color.rgb = rgb(SUAVE)

    def h(codigo, titulo):
        p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(7); p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        txt(p, codigo + " ", 12, ROJO, bold=True); txt(p, "→ " + titulo.upper(), 9.5, ROJO, bold=True)

    def nota(texto):
        p = doc.add_paragraph(); txt(p, texto, 8.5, SUAVE, italic=True); p.paragraph_format.space_after = Pt(6)
        return p

    def tabla(cab, filas, anchos, num_desde=1, size=8.5, estilos=None):
        t = doc.add_table(rows=1 + len(filas), cols=len(cab)); bordes(t); t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        anchos = [Emu(int(w)) for w in anchos]
        for j, w in enumerate(anchos):
            t.columns[j].width = w
        for i, fila in enumerate([cab] + filas):
            trPr = t.rows[i]._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit"); trPr.append(cs)
            if i == 0:
                th = OxmlElement("w:tblHeader"); trPr.append(th)
            estilo = (estilos[i - 1] if (estilos and i) else None)
            if estilo == "banda":
                c = t.cell(i, 0).merge(t.cell(i, len(cab) - 1)); c.text = ""
                for extra in c.paragraphs[1:]:          # al fusionar, Word conserva un párrafo vacío por celda
                    extra._p.getparent().remove(extra._p)
                p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
                txt(p, str(fila[0]).upper(), size, fila[1] if len(fila) > 1 else ROJO, bold=True)
                sombra(c, fila[2] if len(fila) > 2 else "FBE0D6"); continue
            for j, v in enumerate(fila):
                c = t.cell(i, j); c.width = anchos[j]; c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
                if i and j >= num_desde:
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                txt(p, "" if v is None else str(v), size, bold=(i == 0 or estilo == "resumen" or (j == 0 and estilo != "texto")))
                if i == 0:
                    sombra(c, "D9D9D9")
                elif estilo == "resumen":
                    sombra(c, "F2F2F2")
                elif isinstance(estilo, dict) and j in estilo:
                    sombra(c, estilo[j])
        sep = doc.add_paragraph(); sep.paragraph_format.space_after = Pt(0)
        sep.paragraph_format.line_spacing = 0.6
        return t

    color_estado = {"OK": "E5F0DC", "VÁLIDO": "E5F0DC", "REVISAR": "FDF1CF", "PENDIENTE": "EDEDED", "NO VÁLIDO": "FBE9E9"}
    fmt = lambda v, d=1: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else _es(v, d)

    # Cabecera y pie (como el resto de documentos de la asignatura)
    hdr = sec.header; ht = hdr.add_table(rows=1, cols=2, width=util); bordes(ht, "nil")
    ht.cell(0, 0).width = Cm(7); ht.cell(0, 1).width = util - Cm(7)
    ht.cell(0, 0).paragraphs[0].add_run().add_picture(_logo(), width=Cm(4.2))
    p = ht.cell(0, 1).paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    txt(p, AUTORIA[0], 8.5, ROJO, bold=True); txt(p, AUTORIA[1], 8.5, bold=True, italic=True)
    p2 = ht.cell(0, 1).add_paragraph(); p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT; txt(p2, DEPARTAMENTO, 8.5, italic=True)
    if hdr.paragraphs and not hdr.paragraphs[0].text:
        hdr.paragraphs[0]._p.getparent().remove(hdr.paragraphs[0]._p)
    ahora = _dt.datetime.now()
    fp = sec.footer.paragraphs[0]
    txt(fp, f"Reporte Hoja D · Miga · {PRACTICA} · generado por el aplicativo el {ahora:%d/%m/%Y %H:%M} · Página ", 7.5, SUAVE)
    campo(fp, "PAGE"); txt(fp, " de ", 7.5, SUAVE); campo(fp, "NUMPAGES")

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(2)
    txt(p, "REPORTE DE RESULTADOS · HOJA D · MIGA", 15, ROJO, bold=True)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(2)
    txt(p, f"Práctica {PRACTICA} → salidas del aplicativo de análisis alveolar", 10.5, bold=True)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(8)
    txt(p, f"Grupos: {', '.join(grupos) if grupos else 'ninguno'} · Resolución de análisis {_es(RES_ANALISIS, 0)} px/mm · "
           f"Zona analizada {_es(LADO_ROI_MM, 0)} × {_es(LADO_ROI_MM, 0)} mm", 9, SUAVE, italic=True)
    nota("Salidas del aplicativo en el orden de la Hoja D · Miga. El % grandes (100 − finos − medios) y la interpretación "
         "se hacen en la Hoja D. «—» = dato no medido: en la Hoja D se escribe NM.")

    # D.1 · Blanco
    h("D.1", "Blanco control")
    if blanco:
        nc = _nombres_clase()
        fila = ["Blanco (cartulina gris)", fmt(blanco.get("escala_px_mm"), 2), fmt(blanco.get("area_alveolar_%")),
                fmt(blanco.get("densidad_alv_cm2")), fmt(blanco.get("feret_p90_mm"), 2),
                fmt(blanco.get(f"% {nc[0]}")), fmt(blanco.get(f"% {nc[1]}"))]
        w0 = Cm(4.2); wr = (util - w0) / 6
        tabla(["Muestra", "Escala (px/mm)", "Área alveolar (%)", "Densidad (alv/cm²)", "Feret P90 (mm)",
               "% < 0,5 mm", "% 0,5–1 mm"], [fila], [w0] + [wr] * 6)
        p = doc.add_paragraph(); txt(p, "Veredicto del blanco: ", 9.5, bold=True)
        r = txt(p, blanco.get("estado", "—"), 9.5, bold=True)
        txt(p, " · " + str(blanco.get("motivo", "")), 9.5)
        txt(p, f"   (criterio: área alveolar ≤ {_es(BLANCO_AREA_OK)} %)", 8.5, SUAVE, italic=True)
    else:
        nota("No se ha analizado el blanco control en esta sesión: en la Hoja D se escribe NM y todas las lecturas "
             "quedan pendientes.")

    # D.2 · Réplicas por producto
    h("D.2", "Miga por réplica · salidas del aplicativo")
    cols = _cols_hoja_d()
    cab = ["Grupo · réplica"] + [c[1] for c in cols]
    filas, estilos = [], []
    bandas = [("4F81BD", "EEF2F6"), (ROJO, "FBE0D6"), ("8F1D1D", "FDF1CF")]
    for k, g in enumerate(grupos):
        d = reg[reg.muestra == g]
        tc, fc = bandas[k % 3]
        filas.append([g, tc, fc]); estilos.append("banda")
        for _, r in d.iterrows():
            filas.append([r.replica] + [fmt(r.get(c), dd) for c, _, dd in cols]); estilos.append("texto")
        filas.append(["Media"] + [fmt(pd.to_numeric(d[c], errors="coerce").mean(), dd) if c in d else "—"
                                  for c, _, dd in cols]); estilos.append("resumen")
        filas.append(["Recorrido (mín–máx)"] + [
            (f"{fmt(pd.to_numeric(d[c], errors='coerce').min(), dd)}–{fmt(pd.to_numeric(d[c], errors='coerce').max(), dd)}"
             if c in d else "—") for c, _, dd in cols]); estilos.append("resumen")
    if grupos:
        w0 = Cm(3.0); wr = (util - w0) / len(cols)
        tabla(cab, filas, [w0] + [wr] * len(cols), size=8, estilos=estilos)
        nota(f"Escala = resolución de la foto original (todas se analizan a {_es(RES_ANALISIS, 0)} px/mm). % finos y % medios: "
             f"por número de alvéolos; % del área en alvéolos > {_es(LIM_MUY_GRANDES_MM, 0)} mm: por área.")
    else:
        nota("No hay grupos analizados.")

    # D.3 · Validación por grupo
    h("D.3", "Validación y homogeneidad por grupo")
    est_b = blanco.get("estado", "PENDIENTE") if blanco else "PENDIENTE"
    filas, estilos = [], []
    for g in grupos:
        d = reg[reg.muestra == g]
        e = _peor([str(d.estado_sin_blanco.iloc[0]), est_b])
        a = pd.to_numeric(d["area_alveolar_%"], errors="coerce")
        cv = 100 * a.std(ddof=1) / a.mean() if len(a.dropna()) > 1 and a.mean() else np.nan
        rv = d["controles_revisar"].dropna().astype(str) if "controles_revisar" in d else pd.Series(dtype=str)
        rv = rv.iloc[0] if len(rv) and rv.iloc[0].strip() and rv.iloc[0] != "nan" else "ninguno"
        if est_b != "OK" and est_b != "VÁLIDO":
            rv = ("Blanco control; " + rv) if rv != "ninguno" else "Blanco control"
        filas.append([g, e, rv, fmt(cv)]); estilos.append({1: color_estado.get(e, "FFFFFF")})
    if grupos:
        tabla(["Grupo", "Estado de validación", "Controles en REVISAR", "CV del área alveolar (%)"], filas,
              [Cm(3.0), Cm(3.4), util - Cm(9.6), Cm(3.2)], num_desde=3, size=8.5, estilos=estilos)
        nota("CV (%) = desviación típica de las réplicas ÷ media × 100. Detalle de cada control: hoja «Validacion» del "
             "Excel del grupo.")

    # D.4 · Comparación
    h("D.4", "Comparación entre grupos · lectura del aplicativo")
    if comparar and len(grupos) >= 2:
        import matplotlib.pyplot as _plt
        r = comparar_grupos(reg, blanco, grupos, None)
        for f in r["figs"].values():
            if f is not None:
                _plt.close(f)
        pares = r["pares"]
        variables = [dict(VARS_COMP)[v] for v in ("area_alveolar_%", "densidad_alv_cm2", "%_area_gt2mm")]
        filas, estilos = [], []
        for i, a in enumerate(grupos):
            for b in grupos[i + 1:]:
                fila = [f"{a} – {b}"]; col = {}
                for j, var in enumerate(variables, start=1):
                    x = pares[(pares.Variable == var) & (((pares["Grupo A"] == a) & (pares["Grupo B"] == b))
                                                          | ((pares["Grupo A"] == b) & (pares["Grupo B"] == a)))]
                    if len(x):
                        lec = str(x.Lectura.iloc[0])
                        fila.append(f"{lec} · recorridos {x.Recorridos.iloc[0]} · p = {fmt(x['Tukey p ajustado'].iloc[0], 3)}")
                        col[j] = ("E5F0DC" if lec.startswith("Diferencia sólida") else
                                  "FDF1CF" if lec.startswith("Indicio") else "FFFFFF")
                    else:
                        fila.append("—")
                filas.append(fila); estilos.append(col)
        nota("p = Tukey HSD ajustado. Diferencia sólida = recorridos sin solape y p < 0,05. Indicio = solo uno de los dos "
             "criterios. «No se puede afirmar que difieran» no significa «son iguales».")
        tabla(["Par", "Área alveolar", "Densidad alveolar", f"% del área en alvéolos > {_es(LIM_MUY_GRANDES_MM, 0)} mm"],
              filas, [Cm(3.2)] + [(util - Cm(3.2)) / 3] * 3, num_desde=9, size=8, estilos=estilos)
    elif comparar:
        nota("Hacen falta al menos 2 grupos para la comparación.")
    else:
        nota("La comparación entre grupos se incluye en el reporte de la descarga del paso 5, con todos los grupos.")
    ult = doc.paragraphs[-1]                      # el párrafo final obligatorio no debe abrir una página vacía
    if not ult.text:
        ult.paragraph_format.line_spacing = Pt(1); ult.paragraph_format.space_after = Pt(0)
    z = doc.settings.element.find(qn("w:zoom"))
    if z is not None and z.get(qn("w:percent")) is None:
        z.set(qn("w:percent"), "100")
    doc.core_properties.title = "Reporte Hoja D · Miga"; doc.core_properties.author = AUTORIA[1]
    doc.save(ruta)
    return ruta


# ============================================================
#  APLICATIVO (interfaz)
# ============================================================
import re
import shutil
import ipywidgets as W
from IPython.display import display, HTML

AZUL = "#2a78d6"
OPC_ZONA = ["Cuadrado central de 24 mm (recomendado)", "Cuadrado central de 20 mm (si los bordes salen oscuros)",
            "Cuadrado central de 26 mm"]
OPC_UMBRAL = ["Otsu (recomendado)", "Li (solo para comprobar)", "Media (solo para comprobar)"]
OPC_MIN = ["0,2 mm (recomendado)", "0,15 mm", "0,3 mm"]
OPC_CONTROL = ["Solo si hay avisos", "Siempre", "Nunca"]
EJEMPLO = "AF_R1; AF_R2; AF_R3"


# Todo el HTML propio lleva fondo y color explícitos: se lee igual con el tema claro y con el oscuro de Colab.
def _cabecera():
    return ("<div style='background:linear-gradient(90deg,#1c5cab,#2a78d6);color:#ffffff;border-radius:10px;"
            "padding:16px 20px;margin:4px 0 12px'><div style='font-size:22px;font-weight:700;color:#ffffff'>"
            f"Análisis alveolar de miga · {PRACTICA}</div><div style='font-size:13px;color:#e8f0fb;margin-top:2px'>"
            "Bizcochos de espuma · Panificación y Pastelería · área alveolar, densidad y tamaños de 3 réplicas, "
            "con validación por blanco control y por réplicas</div></div>")


def _paso(n, titulo):
    return (f"<div style='background:#eef4fc;color:#1c3f73;border-radius:8px;padding:8px 12px;margin:12px 0 8px;"
            f"display:flex;align-items:center;gap:10px'><span style='background:{AZUL};color:#ffffff;"
            f"border-radius:50%;width:26px;height:26px;display:inline-flex;align-items:center;"
            f"justify-content:center;font-weight:700'>{n}</span><span style='font-size:16px;font-weight:600;"
            f"color:#1c3f73'>{titulo}</span></div>")


def _tile(valor, etiqueta, detalle="", estado=None):
    fondo, tinta = ("#ffffff", "#1f2328") if estado is None else ESTILO_ESTADO[estado][:2]
    return (f"<div style='flex:1;min-width:150px;border:1px solid #d9d7cf;border-radius:8px;padding:10px 14px;"
            f"background:{fondo}'><div style='font-size:12px;color:#5f5e58'>{etiqueta}</div>"
            f"<div style='font-size:24px;font-weight:700;color:{tinta};line-height:1.25'>{valor}</div>"
            f"<div style='font-size:12px;color:#5f5e58'>{detalle}</div></div>")


def _insignia(estado, texto=None):
    f, t, i = ESTILO_ESTADO[estado]
    return (f"<span style='background:{f};color:{t};border-radius:4px;padding:1px 7px;font-weight:700;"
            f"font-size:12px;white-space:nowrap'>{i} {texto or estado}</span>")


def _nombre_muestra(etiquetas):
    """Nombre común de las réplicas (AF_R1; AF_R2 -> AF)."""
    if not etiquetas:
        return "Muestra"
    pref = os.path.commonprefix(etiquetas).rstrip(" _-.")
    pref = re.sub(r"[ _\-.]+[A-Za-z]{1,2}$", "", pref).rstrip(" _-.")   # AF_R -> AF
    return pref if len(pref) >= 2 else "Pieza"


def _html_sesion():
    """Recuadro con el estado de validación de la sesión: blanco y piezas registradas."""
    b = blanco_cargar(); reg = registro_cargar()
    if b:
        tb = (_insignia(b["estado"], "VÁLIDO" if b["estado"] == "OK" else b["estado"])
              + f" área alveolar {_es(b['area_alveolar_%'])} % · {b['fecha']}")
    else:
        tb = _insignia("PENDIENTE") + " sin analizar: fotografíe la cartulina gris y analícela como «Blanco control»"
    eb = b["estado"] if b else "PENDIENTE"
    piezas = []
    for m in dict.fromkeys(reg.muestra):
        s = reg[reg.muestra == m]
        piezas.append(f"<b>{m}</b> ({len(s)} réplicas) {_insignia(_peor([s.estado_sin_blanco.iloc[0], eb]))}")
    tp = " · ".join(piezas) if piezas else "ninguna todavía"
    return (f"<div style='background:#f6f5f1;color:#1f2328;border-radius:8px;padding:8px 12px;margin:0 0 6px;"
            f"font-size:13px;line-height:1.7'><b>Validación de la sesión</b><br>Blanco control: {tb}<br>"
            f"Grupos analizados: {tp}</div>")


TIPO_BLANCO, TIPO_PIEZA = "Blanco control (1 foto)", "Grupo de muestras (3 réplicas)"
EJEMPLO_GRUPOS = "AF · SEP · GENOV-M · GENOV-SM"


def _archivos_subidos(w):
    """Archivos de un W.FileUpload como [(nombre, bytes)]. Compatible con ipywidgets 7 (Colab) y 8."""
    v = w.value
    if isinstance(v, dict):                                    # ipywidgets 7
        return [(k, bytes(d["content"])) for k, d in v.items()]
    return [(d["name"], bytes(d["content"])) for d in v]       # ipywidgets 8


def _vaciar_subida(w):
    try:
        w.value = ()                                           # ipywidgets 8
    except Exception:
        try:
            w.value.clear(); w._counter = 0                    # ipywidgets 7
        except Exception:
            pass


class AplicativoAlveolar:
    """Flujo de la sesión en un solo panel, sin volver a ejecutar la celda:
    1) cargar fotos (blanco o grupo), 2) nombre del grupo y de cada réplica, 3) analizar, validar
    y descargar; «Analizar otro grupo» vuelve al paso 1 conservando todo lo anterior. No hay límite
    de grupos. 4) comparar los grupos elegidos y descargar los resultados elegidos."""

    def __init__(self, carpeta_fotos="fotos", carpeta_salida="resultados"):
        if os.path.isdir("/content"):
            os.chdir("/content")
        self.fotos_dir, self.salida = carpeta_fotos, carpeta_salida
        shutil.rmtree(self.fotos_dir, ignore_errors=True); os.makedirs(self.fotos_dir)
        os.makedirs(self.salida, exist_ok=True)                # los resultados de la sesión se conservan
        self.zip, self.zip_comp, self._confirmar_reg, self._ocupado = None, None, False, False
        display(HTML(_cabecera()))
        self.sesion = W.HTML(); self.guia = W.HTML()
        display(self.sesion, self.guia)
        self._construir_panel()
        display(self.panel)
        self._construir_comparacion()
        display(self.panel_comp)
        self._reiniciar_muestra(primera=True)

    # ---------- paso 1 · cargar fotos ----------
    @property
    def rutas(self):
        return sorted(f for f in glob.glob(os.path.join(self.fotos_dir, "*")) if f.lower().endswith(EXT_IMG))

    def _guardar_fotos(self, archivos):
        """Sustituye las fotos de la muestra actual. archivos: [(nombre, bytes)] o [ruta]."""
        shutil.rmtree(self.fotos_dir, ignore_errors=True); os.makedirs(self.fotos_dir)
        rechazadas = []
        for a in archivos:
            nombre = os.path.basename(a[0] if isinstance(a, tuple) else a)
            if nombre.lower().endswith((".heic", ".heif")):
                rechazadas.append(f"«{nombre}» está en HEIC (iPhone): expórtela a JPG")
                continue
            if not nombre.lower().endswith(EXT_IMG):
                rechazadas.append(f"«{nombre}» no es una imagen JPG o PNG")
                continue
            destino = os.path.join(self.fotos_dir, nombre)
            if isinstance(a, tuple):
                with open(destino, "wb") as f:
                    f.write(a[1])
            else:
                shutil.move(a, destino)
        return rechazadas

    def _al_subir(self, cambio=None):
        archivos = _archivos_subidos(self.subida)
        if not archivos:
            return
        self._cargar(archivos)
        _vaciar_subida(self.subida)

    def _desde_panel_archivos(self, _=None):
        sueltas = [f for f in os.listdir(".") if f.lower().endswith(EXT_IMG + (".heic", ".heif"))]
        if not sueltas:
            self.carga_estado.value = _nota(
                "No hay fotos sueltas en el panel 📁 <b>Archivos</b>. Arrástrelas allí (fuera de cualquier carpeta) "
                "y vuelva a pulsar este botón.", "aviso")
            return
        self._cargar(sueltas)

    def _cargar(self, archivos):
        rechazadas = self._guardar_fotos(archivos)
        n = len(self.rutas)
        self.mini.clear_output()
        with self.mini:
            self._mostrar_fotos()
        avisos = "".join(_nota("⚠ " + r, "error") for r in rechazadas)
        if n == 0:
            self.carga_estado.value = avisos + _nota("No se ha cargado ninguna foto válida.", "error")
            return
        self.rep_txt, self._auto = [], []          # fotos nuevas: casillas de réplica nuevas
        self._construir_replicas()
        # El número de fotos decide el tipo por defecto; la persona puede cambiarlo.
        self.tipo.value = TIPO_BLANCO if n == 1 else TIPO_PIEZA
        esperado = 1 if self._es_blanco() else 3
        tipo_txt = "del blanco control" if self._es_blanco() else "del grupo de muestras"
        self.carga_estado.value = avisos + (
            _nota(f"✔ <b>{n} {'foto cargada' if n == 1 else 'fotos cargadas'}</b> {tipo_txt}. Continúe con el paso 2.",
                  "ok") if n == esperado else
            _nota(f"⚠ Se han cargado <b>{n}</b> fotos. Un grupo se analiza con <b>3</b> réplicas: vuelva a elegir "
                  "las fotos seleccionándolas todas a la vez (se puede continuar igualmente).", "aviso"))
        self.caja_datos.layout.display = "flex"
        self._cambiar_tipo()

    def _mostrar_fotos(self):
        rutas = self.rutas
        if not rutas:
            return
        fig, ax = plt.subplots(1, len(rutas), figsize=(3.0 * len(rutas), 3.4), squeeze=False)
        fig.patch.set_facecolor("white")
        for i, r in enumerate(rutas):
            try:
                ax[0, i].imshow(cv2.cvtColor(leer_foto(r), cv2.COLOR_BGR2RGB))
            except Exception:
                ax[0, i].text(0.5, 0.5, "no se puede abrir", ha="center", va="center")
            ax[0, i].axis("off")
            ax[0, i].set_title(f"Foto {i + 1}\n{os.path.basename(r)[:30]}", fontsize=9, color="#1f2328")
        plt.tight_layout(); plt.show()

    # ---------- construcción del panel ----------
    def _construir_panel(self):
        ancho = {"description_width": "initial"}
        # Paso 1
        self.paso1 = W.HTML()
        self.subida = W.FileUpload(accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp,.webp,.heic,.heif", multiple=True,
                                   description="Elegir fotos", button_style="primary",
                                   layout=W.Layout(width="190px", height="40px"))
        self.subida.observe(self._al_subir, names="value")
        self.btn_panel = W.Button(description="Usar fotos del panel Archivos", icon="folder-open",
                                  layout=W.Layout(width="250px", height="40px", margin="0 0 0 10px"))
        self.btn_panel.on_click(self._desde_panel_archivos)
        self.ayuda_carga = W.HTML()
        self.carga_estado = W.HTML()
        self.mini = W.Output()
        # Paso 2
        self.tipo = W.ToggleButtons(options=[TIPO_BLANCO, TIPO_PIEZA], value=TIPO_PIEZA, description="Tipo:",
                                    style={"description_width": "initial", "button_width": "190px"})
        self.tipo.observe(lambda _: self._cambiar_tipo(), "value")
        self.grupo = W.Text(description="Nombre del grupo:", placeholder=f"p. ej. {EJEMPLO_GRUPOS}",
                            style=ancho, layout=W.Layout(width="460px"))
        self.grupo.observe(lambda _: self._nombres_desde_grupo(), "value")
        self.caja_rep = W.VBox()
        self.rep_txt, self._auto = [], []
        self.ayuda_nombres = W.HTML()
        self.asignacion = W.HTML()
        self.zona = W.Dropdown(options=OPC_ZONA, value=OPC_ZONA[0], description="Zona de análisis:",
                               style=ancho, layout=W.Layout(width="560px"))
        self.escala = W.Text(value="", placeholder="vacío = automática con la regla; o p. ej. 31,6 px/mm",
                             description="Escala:", style=ancho, layout=W.Layout(width="560px"))
        self.umbral = W.Dropdown(options=OPC_UMBRAL, value=OPC_UMBRAL[0], description="Umbral:",
                                 style=ancho, layout=W.Layout(width="560px"))
        self.minimo = W.Dropdown(options=OPC_MIN, value=OPC_MIN[0], description="Alvéolo mínimo contado:",
                                 style=ancho, layout=W.Layout(width="560px"))
        self.control = W.Dropdown(options=OPC_CONTROL, value=OPC_CONTROL[0],
                                  description="Imágenes de control de la regla:", style=ancho,
                                  layout=W.Layout(width="560px"))
        self.acordeon = W.Accordion(children=[W.VBox([
            W.HTML(_nota("Las opciones están en el valor recomendado para miga de bizcocho. Si las cambia, cámbielas "
                         "igual en el blanco y en todos los grupos: solo son comparables grupos analizados con las "
                         "mismas opciones.", "aviso")),
            self.zona, self.escala, self.umbral, self.minimo, self.control])])
        self.acordeon.set_title(0, "Opciones avanzadas (ya tienen el valor recomendado)")
        self.acordeon.selected_index = None
        # Paso 3
        self.btn_analizar = W.Button(description="Analizar", icon="play", button_style="primary",
                                     layout=W.Layout(width="190px", height="40px"))
        self.btn_analizar.on_click(self._analizar)
        self.progreso = W.IntProgress(min=0, max=3, value=0, bar_style="info",
                                      layout=W.Layout(width="320px", visibility="hidden", margin="0 0 0 12px"))
        self.estado = W.HTML()
        self.indicadores = W.HTML()
        self.resultados = W.Output()
        self.btn_descargar = W.Button(description="Descargar resultados (ZIP)", icon="download",
                                      button_style="success", layout=W.Layout(width="250px", height="40px",
                                                                              display="none"))
        self.btn_descargar.on_click(lambda _: self._descargar(self.zip, self.salida_extra))
        self.btn_siguiente = W.Button(description="Analizar otro grupo", icon="plus",
                                      button_style="primary",
                                      layout=W.Layout(width="260px", height="40px", display="none", margin="0 0 0 10px"))
        self.btn_siguiente.on_click(lambda _: self._reiniciar_muestra())
        self.salida_extra = W.Output()

        self.caja_datos = W.VBox([
            W.HTML(_paso(2, "Tipo de muestra y nombres")), self.tipo, self.ayuda_nombres, self.grupo,
            W.HTML("<div style='font-size:13px;color:#1f2328;background:#ffffff;margin:6px 0 2px'><b>Nombre de cada "
                   "réplica</b> (en el orden de las fotos; se puede cambiar):</div>"),
            self.caja_rep, self.asignacion, self.acordeon,
            W.HTML(_paso(3, "Análisis, validación y resultados")),
            W.HBox([self.btn_analizar, self.progreso], layout=W.Layout(align_items="center")),
            self.estado, self.indicadores, self.resultados,
            W.HBox([self.btn_descargar, self.btn_siguiente], layout=W.Layout(margin="10px 0 0")),
            self.salida_extra], layout=W.Layout(display="none"))
        self.panel = W.VBox([
            self.paso1, self.ayuda_carga,
            W.HBox([self.subida, self.btn_panel], layout=W.Layout(align_items="center", flex_flow="row wrap")),
            self.carga_estado, self.mini, self.caja_datos], layout=W.Layout(max_width="1150px"))

    def _reiniciar_muestra(self, primera=False):
        """Vuelve al paso 1 para la siguiente muestra. Conserva resultados, ZIP y registro de la sesión."""
        shutil.rmtree(self.fotos_dir, ignore_errors=True); os.makedirs(self.fotos_dir)
        plt.close("all")
        _vaciar_subida(self.subida)
        self.zip = None
        for w in (self.mini, self.resultados, self.salida_extra):
            w.clear_output()
        self.indicadores.value = self.carga_estado.value = self.asignacion.value = ""
        self.grupo.value = ""; self.caja_rep.children = (); self.rep_txt, self._auto = [], []
        self.btn_descargar.layout.display = self.btn_siguiente.layout.display = "none"
        self.btn_analizar.description = "Analizar"
        self.caja_datos.layout.display = "none"
        sin_blanco = blanco_cargar() is None
        self.tipo.value = TIPO_BLANCO if sin_blanco else TIPO_PIEZA
        hechos = list(dict.fromkeys(registro_cargar().muestra))
        if sin_blanco:
            titulo = "Cargue la foto del blanco control"
            texto = ("Empiece por el <b>blanco control</b>: una foto de la cartulina gris dentro de la plantilla, con "
                     "la regla, la misma luz y la misma distancia que usará para las muestras. Pulse <b>Elegir "
                     "fotos</b> y seleccione esa foto.")
        else:
            titulo = f"Cargue las 3 fotos del grupo {len(hechos) + 1}"
            texto = ("Pulse <b>Elegir fotos</b> y seleccione <b>a la vez</b> las 3 fotos (réplicas) de un grupo de "
                     "muestras. Después escribirá el nombre del grupo y el de cada réplica. Puede añadir tantos "
                     "grupos como necesite. Use el archivo original de la cámara: sin recortar, sin captura de "
                     "pantalla y sin pasar por mensajería.")
        self.paso1.value = _paso(1, titulo)
        self.ayuda_carga.value = _nota(texto + " Si el botón no responde, arrastre las fotos al panel 📁 "
                                       "<b>Archivos</b> de la izquierda y pulse <b>Usar fotos del panel Archivos</b>.",
                                       "info")
        if not primera:
            self.carga_estado.value = _nota("Resultados anteriores guardados: siguen en el registro de la sesión "
                                            "y se pueden descargar en el paso 5.", "ok")
        self._refrescar()

    # ---------- paso 2 · tipo y nombres ----------
    def _es_blanco(self):
        return self.tipo.value == TIPO_BLANCO

    def _construir_replicas(self):
        """Una casilla de nombre por foto, en el orden de las fotos."""
        rutas = self.rutas; previos = [t.value for t in self.rep_txt]
        self.rep_txt, filas = [], []
        for i, r in enumerate(rutas):
            t = W.Text(value=previos[i] if i < len(previos) else "", placeholder=f"réplica {i + 1}",
                       layout=W.Layout(width="260px"))
            t.observe(lambda _: self._actualizar(), "value")
            self.rep_txt.append(t)
            filas.append(W.HBox([W.HTML(f"<div style='width:300px;font-size:13px;color:#5f5e58;background:#ffffff;"
                                        f"padding-top:5px'>Foto {i + 1} · {os.path.basename(r)[:34]}</div>"), t]))
        self._auto = self._auto[:len(rutas)] + [""] * max(0, len(rutas) - len(self._auto))
        self.caja_rep.children = tuple(filas)

    def _cambiar_tipo(self):
        n = len(self.rutas)
        if len(self.rep_txt) != n:
            self._construir_replicas()
        if self._es_blanco():
            self.ayuda_nombres.value = _nota(
                "<b>Blanco control:</b> cartulina gris mate fotografiada con la misma plantilla, la misma regla, la "
                "misma luz y la misma distancia que las muestras. Debe dar un área alveolar próxima a cero. Se hace "
                "<b>al principio de la sesión</b> y se repite si cambia la luz o la cámara.", "info")
            self.grupo.layout.display = "none"
            for i, t in enumerate(self.rep_txt):
                t.value = "Blanco" if n == 1 else f"Blanco_{i + 1}"
            self._auto = [t.value for t in self.rep_txt]
        else:
            self.ayuda_nombres.value = _nota(
                f"Escriba el <b>nombre del grupo de muestras</b> (p. ej. {EJEMPLO_GRUPOS}). Cada réplica recibe "
                "un nombre provisional <b>GRUPO_R1, _R2, _R3</b> que puede cambiar en su casilla.", "info")
            self.grupo.layout.display = "flex"
            for i, t in enumerate(self.rep_txt):
                if t.value.lower().startswith("blanco"):
                    t.value = ""
            self._nombres_desde_grupo()
        self._actualizar()

    def _nombres_desde_grupo(self):
        """Propone GRUPO_R1… sin pisar los nombres que la persona haya escrito a mano."""
        if self._es_blanco():
            return
        g = re.sub(r"[;\s]+", "_", self.grupo.value.strip()).strip("_")
        for i, t in enumerate(self.rep_txt):
            nuevo = f"{g}_R{i + 1}" if g else ""
            if t.value.strip() in ("", self._auto[i]):
                t.value = nuevo
            self._auto[i] = nuevo
        self._actualizar()

    def _etiquetas(self):
        return [t.value.strip() for t in self.rep_txt]

    def _nombre_grupo(self):
        return "Blanco" if self._es_blanco() else self.grupo.value.strip()

    def _actualizar(self):
        if not hasattr(self, "btn_analizar"):
            return
        n, et, g = len(self.rutas), self._etiquetas(), self._nombre_grupo()
        vacios = [i + 1 for i, e in enumerate(et) if not e]
        repes = sorted({e for e in et if e and et.count(e) > 1})
        ok = n > 0 and bool(g) and not vacios and not repes and len(et) == n
        self.asignacion.value = ""
        if self.zip or self._ocupado:
            return
        if not g:
            self.estado.value = _nota("Escriba el <b>nombre del grupo</b> de muestras en el paso 2.", "info")
        elif vacios:
            self.estado.value = _nota(f"Falta el nombre de la réplica de la foto {', '.join(map(str, vacios))}.", "aviso")
        elif repes:
            self.estado.value = _nota(f"Nombres de réplica repetidos: <b>{', '.join(repes)}</b>. Cada réplica debe "
                                      "tener un nombre distinto.", "aviso")
        elif self._es_blanco():
            ya = blanco_cargar() is not None
            self.estado.value = _nota("Todo listo para analizar el <b>blanco control</b>. Pulse <b>Analizar</b>."
                                      + (" <i>Sustituirá al blanco anterior de la sesión.</i>" if ya else ""), "ok")
        else:
            ya = g in set(registro_cargar().muestra)
            self.estado.value = _nota(f"Todo listo. Grupo <b>{g}</b> con réplicas <b>{'; '.join(et)}</b>. Pulse "
                                      "<b>Analizar</b>." + (" <i>Ya hay un grupo con ese nombre en la sesión: se "
                                                           "sustituirá.</i>" if ya else ""), "ok")
        self.btn_analizar.disabled = not ok

    # ---------- paso 3 · análisis ----------
    def _avance(self, i, n, etq):
        self.progreso.max = n; self.progreso.value = i
        self.estado.value = _nota(f"⏳ Analizando {etq} ({i + 1} de {n})" if etq != "fin"
                                  else "⏳ Generando figuras, tablas, validación y documentos", "info")

    def _analizar(self, _=None):
        et = self._etiquetas()
        opciones = interpretar_opciones(self.zona.value, self.umbral.value, self.minimo.value,
                                        self.escala.value or "auto")
        bloquear = (self.btn_analizar, self.grupo, self.tipo, self.subida, self.btn_panel, *self.rep_txt)
        for w in bloquear:
            w.disabled = True
        self._ocupado = True
        self.progreso.layout.visibility = "visible"
        self.resultados.clear_output(); self.indicadores.value = ""; self.zip = None
        self.btn_descargar.layout.display = self.btn_siguiente.layout.display = "none"
        # El análisis se ejecuta DENTRO del panel de resultados y captura allí sus propios errores:
        # si una excepción llegara a salir del panel, Colab mostraría el traceback en lugar del aviso.
        error, hecho, era_blanco = None, "", self._es_blanco()
        try:
            with self.resultados:
                try:
                    if era_blanco:
                        blanco, self.zip = analizar_blanco(self.rutas, opciones,
                                                           salida=os.path.join(self.salida, "Blanco_control"),
                                                           mostrar_control=self.control.value,
                                                           al_avanzar=self._avance, etiquetas=et)
                        self._indicadores_blanco(blanco)
                        hecho = "Blanco control analizado y registrado para la sesión"
                    else:
                        muestra = self._nombre_grupo()
                        tabla, self.zip, est = analizar_muestra(self.rutas, muestra, opciones,
                                                                salida=os.path.join(self.salida, _archivo_seguro(muestra)),
                                                                mostrar_control=self.control.value,
                                                                al_avanzar=self._avance, etiquetas=et)
                        self._indicadores(tabla, muestra, est)
                        hecho = f"Análisis de «{muestra}» terminado"
                except Exception as e:
                    error = e
            if error is not None:
                raise error
            n_g = registro_cargar().muestra.nunique()
            despues = ("Pulse <b>Analizar otro grupo</b> para cargar las 3 fotos del siguiente grupo de muestras."
                       + (f" Con {n_g} grupos ya puede compararlos en el paso 4." if n_g >= 2 else ""))
            self.estado.value = _nota(f"✔ {hecho}. {despues} Los resultados no se pierden: el ZIP de cada muestra "
                                      "se puede descargar ahora o seleccionarlo en el paso 5.", "ok")
            self.btn_descargar.layout.display = "flex"; self.btn_siguiente.layout.display = "flex"
            self.btn_analizar.description = "Volver a analizar"
        except Exception as e:
            self.estado.value = _nota(f"✖ No se pudo completar el análisis: {e} Puede cambiar las fotos con "
                                      "<b>Elegir fotos</b> (paso 1) y volver a analizar.", "error")
        finally:
            self._ocupado = False
            self.progreso.layout.visibility = "hidden"
            for w in bloquear:
                w.disabled = False
            self._refrescar()

    def _indicadores(self, t, muestra, estado):
        t = t.set_index("Réplica")
        fila = "Media" if "Media" in t.index else t.index[0]
        de = t.loc["DE"] if "DE" in t.index else None
        rec = (f"recorrido {_es(t.loc['Mín', 'Área alveolar (%)'])}-{_es(t.loc['Máx', 'Área alveolar (%)'])} %"
               if "Mín" in t.index else "de la superficie analizada")
        dens_det = f"± {_es(de['Densidad (alv/cm²)'])} alv/cm² (DE)" if de is not None else "alv/cm²"
        cl = " · ".join(_es(t.loc[fila, c], 0) for c in _cols_pct()) + " %"
        self.indicadores.value = (
            f"<div style='background:#f6f5f1;border-radius:10px;padding:10px 12px;margin:6px 0'>"
            f"<div style='font-size:13px;color:#5f5e58;margin:0 0 6px'>Resumen de «{muestra}» "
            f"(media de {sum(i not in FILAS_RESUMEN for i in t.index)} réplicas)</div>"
            f"<div style='display:flex;gap:10px;flex-wrap:wrap'>"
            + _tile(f"{_es(t.loc[fila, 'Área alveolar (%)'])} %", "Área alveolar · variable primaria", rec)
            + _tile(_es(t.loc[fila, 'Densidad (alv/cm²)']), "Densidad alveolar", dens_det)
            + _tile(_es(t.loc[fila, 'Feret P90 (mm)'], 2), "Feret P90", "mm (alvéolos más grandes)")
            + _tile(cl, "Finos · medios · grandes", "% del número de alvéolos")
            + _tile(f"{_es(t.loc[fila, '% del área en alvéolos > 2 mm'])} %", "Área en alvéolos > 2 mm",
                    (f"recorrido {_es(t.loc['Mín', '% del área en alvéolos > 2 mm'])}-"
                     f"{_es(t.loc['Máx', '% del área en alvéolos > 2 mm'])} % · huecos grandes")
                    if "Mín" in t.index else "% del área alveolar · huecos grandes")
            + _tile(estado, "Validación", "detalle en la tabla de validación", estado)
            + "</div></div>")

    def _indicadores_blanco(self, b):
        self.indicadores.value = (
            "<div style='background:#f6f5f1;border-radius:10px;padding:10px 12px;margin:6px 0'>"
            "<div style='display:flex;gap:10px;flex-wrap:wrap'>"
            + _tile(f"{_es(b['area_alveolar_%'])} %", "Área alveolar del blanco",
                    f"criterio ≤ {_es(BLANCO_AREA_OK)} %")
            + _tile(_es(b["contraste_DE"]), "Grano de la cartulina (DE)", "niveles de gris")
            + _tile("VÁLIDO" if b["estado"] == "OK" else b["estado"], "Blanco control", b["fecha"], b["estado"])
            + "</div></div>")

    def _descargar(self, ruta, salida):
        salida.clear_output()
        with salida:
            if not ruta:
                return
            try:
                from google.colab import files
                files.download(ruta)
                _html(f"⬇ Descarga iniciada: <b>{os.path.basename(ruta)}</b>", "ok")
            except Exception:
                _html(f"Descargue el archivo desde el panel 📁 Archivos → <b>{ruta}</b>.", "info")

    def _actualizar_guia(self):
        """Recorrido de la sesión: qué está hecho y qué falta."""
        b = blanco_cargar(); hechos = list(dict.fromkeys(registro_cargar().muestra))
        def item(ok, texto):
            col, ic = ("#1b6b3a", "✔") if ok else ("#5f5e58", "○")
            return f"<span style='color:{col};margin-right:16px'>{ic} {texto}</span>"
        grupos = (f"Grupos analizados: {len(hechos)} ({', '.join(hechos)})" if hechos else "Grupos analizados: 0")
        pasos = [item(b is not None, "Blanco control"), item(len(hechos) >= 2, grupos),
                 item(self.zip_comp is not None, "Comparación")]
        self.guia.value = (f"<div style='background:#ffffff;color:#1f2328;border:1px solid #d9d7cf;border-radius:8px;"
                           f"padding:6px 12px;margin:0 0 6px;font-size:13px;line-height:1.6'><b>Recorrido de la "
                           f"sesión:</b> {''.join(pasos)}</div>")

    # ---------- 4 · comparación entre grupos y descargas ----------
    def _construir_comparacion(self):
        self._sel_comp, self._sel_desc = {}, {}
        self.comp_lista = W.HTML()
        self.caja_sel_comp = W.Box(layout=W.Layout(flex_flow="row wrap"))
        self.btn_comparar = W.Button(description="Comparar grupos seleccionados", icon="balance-scale",
                                     button_style="primary", layout=W.Layout(width="260px", height="38px"))
        self.btn_comparar.on_click(self._comparar)
        self.btn_vaciar = W.Button(description="Vaciar registro", icon="trash",
                                   layout=W.Layout(width="170px", height="38px", margin="0 0 0 10px"))
        self.btn_vaciar.on_click(self._vaciar)
        self.comp_estado = W.HTML()
        self.comp_out = W.Output()
        self.caja_sel_desc = W.Box(layout=W.Layout(flex_flow="row wrap"))
        self.btn_desc_sel = W.Button(description="Descargar seleccionados (un ZIP)", icon="download",
                                     button_style="success", layout=W.Layout(width="280px", height="38px"))
        self.btn_desc_sel.on_click(self._descargar_seleccion)
        self.importar = W.FileUpload(accept=".zip,.xlsx", multiple=True, description="Importar resultados anteriores",
                                     layout=W.Layout(width="270px", height="34px"))
        self.importar.observe(self._importar, names="value")
        self.desc_out = W.Output()
        self.panel_comp = W.VBox([
            W.HTML(_paso(4, "Comparación entre grupos")),
            W.HTML(_nota("Se comparan todos los grupos marcados (por defecto, todos los analizados). La variable "
                         "primaria es el <b>área alveolar</b>. Dos criterios, con la réplica como unidad: "
                         "<b>recorridos sin solape</b> (regla del guion) y <b>Tukey HSD</b> (corrige por comparar muchos "
                         "pares a la vez). Una diferencia es <b>sólida</b> cuando la sostienen los dos.", "info")),
            self.comp_lista, self.caja_sel_comp,
            W.HBox([self.btn_comparar, self.btn_vaciar], layout=W.Layout(flex_flow="row wrap", margin="6px 0 0")),
            self.comp_estado, self.comp_out,
            W.HTML(_paso(5, "Descargas")),
            W.HTML(_nota("Marque los resultados que quiere descargar: se entregan juntos en un solo ZIP, con una "
                         "carpeta por muestra. El <b>Reporte Hoja D · Miga</b> (Word) reúne, en el orden de la hoja de "
                         "registro, el blanco, las réplicas con media y recorrido, la validación y la lectura de cada "
                         "par de los grupos marcados.", "info")),
            self.caja_sel_desc, self.btn_desc_sel, self.desc_out,
            W.HTML("<div style='font-size:12px;color:#5f5e58;background:#ffffff;margin:10px 0 4px'>Si la sesión de "
                   "Colab se reinició y perdió el registro, recupérelo con los ZIP que ya descargó:</div>"),
            self.importar], layout=W.Layout(max_width="1150px"))

    def _casillas(self, caja, memoria, opciones, defecto=True):
        """Casillas de verificación que conservan lo marcado entre actualizaciones."""
        cajas = []
        for clave, etiqueta in opciones:
            if clave not in memoria:
                memoria[clave] = W.Checkbox(value=defecto, description=etiqueta, indent=False,
                                            layout=W.Layout(width="auto", margin="0 18px 0 0"))
                memoria[clave].observe(lambda _: self._estado_botones(), "value")
            memoria[clave].description = etiqueta
            cajas.append(memoria[clave])
        for clave in [k for k in memoria if k not in dict(opciones)]:
            del memoria[clave]
        caja.children = tuple(cajas)

    def _resultados_sesion(self):
        """[(clave, etiqueta, carpeta o archivo)] de todo lo descargable en la sesión."""
        out = []
        b = os.path.join(self.salida, "Blanco_control")
        if glob.glob(os.path.join(b, "*")):
            out.append(("blanco", "Blanco control", b))
        reg = registro_cargar()
        for g in dict.fromkeys(reg.muestra):
            d = os.path.join(self.salida, _archivo_seguro(g))
            out.append((f"g:{g}", f"{g}" + ("" if glob.glob(os.path.join(d, "*")) else " (importado: solo registro)"), d))
        comp = os.path.join(REGISTRO_DIR, "comparacion")
        if glob.glob(os.path.join(comp, "*")):
            out.append(("comp", "Comparación entre grupos", comp))
        if len(reg):
            out.insert(0, ("reporte", "Reporte Hoja D · Miga (Word, todos los grupos marcados)", None))
        return out

    def _refrescar(self):
        self.sesion.value = _html_sesion()
        self._actualizar_guia()
        if not hasattr(self, "comp_lista"):
            return
        grupos = list(dict.fromkeys(registro_cargar().muestra))
        self.comp_lista.value = _nota(
            f"Grupos guardados en esta sesión: <b>{len(grupos)}</b>. "
            + ("Marque los que quiere comparar." if len(grupos) >= 2 else "Hacen falta al menos 2 para comparar."),
            "ok" if len(grupos) >= 2 else "info")
        self._casillas(self.caja_sel_comp, self._sel_comp, [(g, g) for g in grupos])
        self._casillas(self.caja_sel_desc, self._sel_desc, [(k, e) for k, e, _ in self._resultados_sesion()])
        self._estado_botones()

    def _estado_botones(self):
        self.btn_comparar.disabled = sum(c.value for c in self._sel_comp.values()) < 2
        self.btn_desc_sel.disabled = not any(c.value for c in self._sel_desc.values())

    def _comparar(self, _=None):
        self.comp_out.clear_output(); self.comp_estado.value = ""
        reg = registro_cargar(); blanco = blanco_cargar()
        grupos = [g for g, c in self._sel_comp.items() if c.value]
        dir_c = os.path.join(REGISTRO_DIR, "comparacion")
        shutil.rmtree(dir_c, ignore_errors=True); os.makedirs(dir_c)
        base = os.path.join(dir_c, "comparacion_grupos")
        try:
            error = None
            with self.comp_out:
                try:
                    r = comparar_grupos(reg, blanco, grupos, base)
                    self._mostrar_comparacion(r, blanco)
                except Exception as e_:
                    error = e_
            if error is not None:
                raise error
            self.zip_comp = base + ".zip"
            with zipfile.ZipFile(self.zip_comp, "w") as z:
                for f in r["archivos"]:
                    z.write(f, os.path.basename(f))
            self.comp_estado.value = _nota(f"✔ Comparación de {len(grupos)} grupos hecha. Puede descargarla en el "
                                           "paso 5 («Comparación entre grupos»).", "ok")
        except Exception as e:
            self.comp_estado.value = _nota(f"✖ No se pudo comparar: {e}", "error")
        self._refrescar()

    def _mostrar_comparacion(self, r, blanco):
        info = r["info"]
        if len(r["configs"]) > 1:
            _html("⚠ Los grupos no se analizaron todos con las mismas opciones: " + " | ".join(r["configs"])
                  + ". Solo son comparables los analizados con las mismas.", "aviso")
        if not blanco:
            _html("⚠ Falta el blanco control de la sesión: las lecturas quedan pendientes hasta analizarlo.", "aviso")
        if info.get("resolucion_desigual"):
            _html("⚠ Los grupos no se analizaron a la misma resolución (px/mm): "
                  + ", ".join(f"{g} {_es(x)}" for g, x in info["resoluciones"].items())
                  + ". La densidad y los % por clase dependen de la resolución: compárelos solo entre grupos con la "
                  "misma. El área alveolar es mucho menos sensible.", "aviso")
        pa = info["anova"].get(VAR_PRIMARIA, np.nan)
        _html(f"<b>{info['n_grupos']} grupos · {info['n_pares']} pares.</b> ANOVA del área alveolar: "
              f"p = {_es(pa, 3) if not np.isnan(pa) else 'no calculable'}. Con la regla de recorridos y "
              f"{info['n_pares']} pares, si no hubiera ninguna diferencia real cabría esperar unas "
              f"<b>{_es(info['falsos_esperados'])}</b> «sin solape» por azar (p = 0,10 por par); por eso se exige "
              "también Tukey para hablar de diferencia sólida.<br><b>Alcance:</b> las 3 réplicas son rebanadas de "
              "una misma pieza; las diferencias describen esas piezas de esta sesión, no la receta en general.",
              "info")
        for k, t in (("variables", "Figura 1. Comparación por variable"),
                     ("matriz", "Figura 2. Matriz de pares de la variable primaria"),
                     ("distribucion", "Figura 3. Distribución del área alveolar por tamaño")):
            if r["figs"].get(k) is not None:
                display(HTML(f"<div style='font-weight:700;color:#1f2328;background:#ffffff;margin:10px 0 2px'>{t}</div>"))
                display(r["figs"][k])
                plt.close(r["figs"][k])
        cols = ["Grupo", "n réplicas", "Estado"] + [c for c in r["resumen"].columns
                                                     if c.startswith(("Área alveolar", "D50"))]
        display(HTML("<div style='font-weight:700;color:#1f2328;background:#ffffff;margin:12px 0 4px'>"
                     "Resumen por grupo (variables de área y tamaño; el Excel de la comparación incluye todas)</div>"))
        display(HTML(html_tabla(r["resumen"][cols], col_estado="Estado")))
        prim = r["pares"][r["pares"].Variable == dict(VARS_COMP)[VAR_PRIMARIA]]
        display(HTML("<div style='font-weight:700;color:#1f2328;background:#ffffff;margin:12px 0 4px'>"
                     "Pares · área alveolar (variable primaria)</div>"))
        display(HTML(html_tabla(prim.drop(columns="Variable"))))
        otras = W.Output()
        with otras:
            display(HTML(html_tabla(r["pares"][r["pares"].Variable != dict(VARS_COMP)[VAR_PRIMARIA]])))
        ac = W.Accordion(children=[otras]); ac.set_title(0, "Pares · densidad, D50 y Feret P90 (secundarias)")
        ac.selected_index = None
        display(ac)

    def _descargar_seleccion(self, _=None):
        import datetime as _dt
        elegidos = [x for x in self._resultados_sesion() if self._sel_desc.get(x[0]) and self._sel_desc[x[0]].value]
        if not elegidos:
            return
        os.makedirs(REGISTRO_DIR, exist_ok=True)
        ruta = os.path.join(REGISTRO_DIR, f"resultados_PR3_{_dt.datetime.now():%Y%m%d_%H%M}.zip")
        grupos = [c[2:] for c, _, _ in elegidos if c.startswith("g:")] or list(dict.fromkeys(registro_cargar().muestra))
        with zipfile.ZipFile(ruta, "w", zipfile.ZIP_DEFLATED) as z:
            for clave, etiqueta, origen in elegidos:
                if clave == "reporte":
                    rep = os.path.join(REGISTRO_DIR, "Reporte_Hoja_D_miga.docx")
                    try:
                        reporte_hoja_d(rep, registro_cargar(), blanco_cargar(), grupos)
                        z.write(rep, os.path.basename(rep))
                    except Exception as e:
                        with self.desc_out:
                            _html(f"⚠ No se pudo generar el reporte de la Hoja D ({e}).", "aviso")
                    continue
                carpeta = _archivo_seguro(etiqueta.split(" (")[0])
                if os.path.isfile(origen):
                    z.write(origen, os.path.join(carpeta, os.path.basename(origen)))
                    continue
                for f in sorted(glob.glob(os.path.join(origen, "*"))):
                    if f.endswith(".zip") or os.path.isdir(f):
                        continue
                    z.write(f, os.path.join(carpeta, os.path.basename(f)))
        self._descargar(ruta, self.desc_out)

    def _importar(self, _=None):
        archivos = _archivos_subidos(self.importar)
        if not archivos:
            return
        self.desc_out.clear_output()
        dest = os.path.join(REGISTRO_DIR, "importados"); os.makedirs(dest, exist_ok=True)
        rutas = []
        for nombre, datos in archivos:
            r = os.path.join(dest, os.path.basename(nombre))
            with open(r, "wb") as f:
                f.write(datos)
            rutas.append(r)
        _vaciar_subida(self.importar)
        piezas, errores = registro_importar(rutas)
        with self.desc_out:
            if piezas:
                _html("✔ Importados: " + ", ".join(dict.fromkeys(piezas)) + ".", "ok")
            for e in errores:
                _html("⚠ " + e, "aviso")
        self._refrescar()

    def _vaciar(self, _=None):
        if not self._confirmar_reg:
            self._confirmar_reg = True
            self.btn_vaciar.description = "¿Seguro? Pulse otra vez"; self.btn_vaciar.button_style = "danger"
            self.comp_estado.value = _nota("Se borrará el registro de la sesión: el blanco y todos los grupos "
                                           "guardados. Descargue antes lo que necesite.", "aviso")
            return
        shutil.rmtree(REGISTRO_DIR, ignore_errors=True); shutil.rmtree(self.salida, ignore_errors=True)
        os.makedirs(self.salida, exist_ok=True)
        self._confirmar_reg = False; self.zip_comp = None
        self.btn_vaciar.description = "Vaciar registro"; self.btn_vaciar.button_style = ""
        self.comp_out.clear_output()
        self.comp_estado.value = _nota("Registro vaciado.", "ok")
        self._reiniciar_muestra(primera=True)


app = AplicativoAlveolar()
