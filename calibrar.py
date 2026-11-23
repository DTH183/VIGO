"""
calibrar.py - Fase A: calibracion de la camara (Logitech C270) con tablero de ajedrez.

Modos:
  python3 calibrar.py            -> tomar fotos (ESPACIO = guardar, C = calibrar, Q = salir)
  python3 calibrar.py calcular   -> calibrar otra vez con las fotos ya guardadas
  python3 calibrar.py probar     -> ver la imagen original vs. corregida (sin distorsion)

No mueve ningun robot. Solo usa la camara.
"""
import glob
import os
import sys

import cv2
import numpy as np

# ---------- Ajustes ----------
CAM_INDEX = 0            # si la C270 no abre, prueba 1
ANCHO, ALTO = 640, 480   # resolucion: SIEMPRE la misma que usaras despues
ESQUINAS = (9, 6)        # esquinas INTERNAS del tablero (10 x 7 cuadros)
SQUARE_MM = 25.0         # <- pon aqui la medida REAL de un cuadro (vernier)
MIN_FOTOS = 15           # minimo de fotos buenas para calibrar

BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA = os.path.join(BASE, "fotos_calib")
ARCHIVO = os.path.join(BASE, "calib_camara.npz")

CRITERIO = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def abrir_camara():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    ok, frame = cap.read()
    if not ok:
        print("No se pudo leer la camara. Revisa el cable o cambia CAM_INDEX.")
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"Resolucion real de la camara: {w} x {h}")
    if (w, h) != (ANCHO, ALTO):
        print(f"OJO: pediste {ANCHO} x {ALTO} pero la camara da {w} x {h}.")
    return cap


def siguiente_nombre():
    os.makedirs(CARPETA, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(CARPETA, f"foto_{n:02d}.png")):
        n += 1
    return os.path.join(CARPETA, f"foto_{n:02d}.png")


def contar_fotos():
    return len(glob.glob(os.path.join(CARPETA, "*.png")))


# ---------- Modo 1: tomar fotos ----------
def tomar_fotos():
    cap = abrir_camara()
    cobertura = []  # esquinas de las fotos guardadas, para ver que zonas ya cubriste
    print("\nESPACIO = guardar foto | C = calibrar | Q = salir")
    print("(Haz clic en la ventana de la camara antes de presionar teclas)\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Fallo la lectura de la camara")
            break
        limpia = frame.copy()                      # esta es la que se guarda
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        encontrado, esquinas = cv2.findChessboardCorners(
            gris, ESQUINAS,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

        # puntos de las fotos ya guardadas (zonas cubiertas)
        for p in cobertura:
            cv2.circle(frame, p, 2, (255, 150, 0), -1)

        if encontrado:
            cv2.drawChessboardCorners(frame, ESQUINAS, esquinas, encontrado)
            texto, color = "TABLERO OK - ESPACIO para guardar", (0, 200, 0)
        else:
            texto, color = "No se ve el tablero completo", (0, 0, 255)

        cv2.putText(frame, texto, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Fotos: {contar_fotos()} (min {MIN_FOTOS}) | C calibrar | Q salir",
                    (10, ALTO - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("calibracion", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord(" "):
            if encontrado:
                nombre = siguiente_nombre()
                cv2.imwrite(nombre, limpia)
                cobertura += [tuple(int(v) for v in c.ravel()) for c in esquinas]
                print("Guardada:", os.path.basename(nombre))
            else:
                print("No guardada: no se detecto el tablero completo")
        elif tecla == ord("c"):
            cap.release()
            cv2.destroyAllWindows()
            calcular()
            return
        elif tecla == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------- Modo 2: calcular la calibracion ----------
def calcular():
    archivos = sorted(glob.glob(os.path.join(CARPETA, "*.png")))
    if not archivos:
        print("No hay fotos en", CARPETA)
        return

    # Puntos 3D del tablero (en mm), el tablero es plano: z = 0
    objp = np.zeros((ESQUINAS[0] * ESQUINAS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:ESQUINAS[0], 0:ESQUINAS[1]].T.reshape(-1, 2) * SQUARE_MM

    pts3d, pts2d, usadas = [], [], []
    tam = None
    for f in archivos:
        gris = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        if tam is None:
            tam = gris.shape[::-1]
        elif gris.shape[::-1] != tam:
            print("Ignorada (otra resolucion):", os.path.basename(f))
            continue
        ok, esquinas = cv2.findChessboardCorners(
            gris, ESQUINAS, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            print("Ignorada (no se detecto el tablero):", os.path.basename(f))
            continue
        esquinas = cv2.cornerSubPix(gris, esquinas, (5, 5), (-1, -1), CRITERIO)
        pts3d.append(objp)
        pts2d.append(esquinas)
        usadas.append(os.path.basename(f))

    print(f"\nFotos validas: {len(usadas)} de {len(archivos)}")
    if len(usadas) < MIN_FOTOS:
        print(f"Necesitas al menos {MIN_FOTOS}. Toma mas fotos con: python3 calibrar.py")
        return

    print("Calculando... (puede tardar unos segundos)")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(pts3d, pts2d, tam, None, None)

    # Error de cada foto, para encontrar las malas
    errores = []
    for i in range(len(usadas)):
        proy, _ = cv2.projectPoints(pts3d[i], rvecs[i], tvecs[i], K, dist)
        e = np.sqrt(np.mean(np.sum((pts2d[i].reshape(-1, 2) - proy.reshape(-1, 2)) ** 2, axis=1)))
        errores.append((e, usadas[i]))
    errores.sort(reverse=True)

    print("\n========== RESULTADO ==========")
    print(f"Resolucion: {tam[0]} x {tam[1]}")
    print(f"Error de reproyeccion (RMS): {rms:.3f} px")
    if rms < 0.5:
        print("  -> EXCELENTE")
    elif rms < 1.0:
        print("  -> ACEPTABLE (se puede mejorar quitando las peores fotos)")
    else:
        print("  -> MALO: borra las peores fotos y/o toma mas, luego: python3 calibrar.py calcular")
    print(f"fx = {K[0,0]:.1f}  fy = {K[1,1]:.1f}   (deben ser parecidos)")
    print(f"cx = {K[0,2]:.1f}  cy = {K[1,2]:.1f}   (cerca del centro: {tam[0]/2:.0f}, {tam[1]/2:.0f})")
    print("Distorsion:", np.round(dist.ravel(), 4))
    print("\nLas 5 fotos con MAS error (candidatas a borrar):")
    for e, n in errores[:5]:
        print(f"  {n}: {e:.3f} px")

    np.savez(ARCHIVO, camera_matrix=K, dist_coeffs=dist, image_size=np.array(tam),
             rms=rms, square_mm=SQUARE_MM, n_fotos=len(usadas))
    print("\nGuardado en:", ARCHIVO)


# ---------- Modo 3: probar la calibracion ----------
def probar():
    if not os.path.exists(ARCHIVO):
        print("Primero calibra: python3 calibrar.py")
        return
    d = np.load(ARCHIVO)
    K, dist = d["camera_matrix"], d["dist_coeffs"]
    cap = abrir_camara()
    print("Izquierda = original | Derecha = corregida | Q = salir")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        corregida = cv2.undistort(frame, K, dist)
        cv2.imshow("original | corregida", np.hstack([frame, corregida]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "fotos"
    if modo == "calcular":
        calcular()
    elif modo == "probar":
        probar()
    else:
        tomar_fotos()
