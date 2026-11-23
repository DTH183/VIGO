"""
medir_distancia.py - Verifica la calibracion: mide la distancia camara -> tablero en mm.

  python3 medir_distancia.py

Teclas (con clic en la ventana):  ESPACIO = imprimir medicion en la terminal | Q = salir
No mueve ningun robot. Necesita calib_camara.npz (de calibrar.py).
"""
import os
import sys
from collections import deque

import cv2
import numpy as np

CAM_INDEX = 0            # el mismo que en calibrar.py
ESQUINAS = (9, 6)        # esquinas internas del tablero
PROMEDIO = 10            # cuadros a promediar para que el numero no brinque

BASE = os.path.dirname(os.path.abspath(__file__))
ARCHIVO = os.path.join(BASE, "calib_camara.npz")
CRITERIO = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def cargar_calibracion():
    if not os.path.exists(ARCHIVO):
        print("No existe calib_camara.npz. Primero corre: python3 calibrar.py")
        sys.exit(1)
    d = np.load(ARCHIVO)
    K, dist = d["camera_matrix"], d["dist_coeffs"]
    ancho, alto = (int(v) for v in d["image_size"])
    sq = float(d["square_mm"])
    print(f"Calibracion cargada: {ancho} x {alto}, cuadro = {sq} mm, RMS = {float(d['rms']):.3f} px")
    return K, dist, ancho, alto, sq


def puntos_tablero(sq):
    objp = np.zeros((ESQUINAS[0] * ESQUINAS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:ESQUINAS[0], 0:ESQUINAS[1]].T.reshape(-1, 2) * sq
    centro = np.array([(ESQUINAS[0] - 1) * sq / 2, (ESQUINAS[1] - 1) * sq / 2, 0.0])
    return objp, centro


def medir(gris, K, dist, objp, centro, rapido=True):
    """Devuelve (Z, distancia, inclinacion, rvec, tvec, esquinas) o None si no ve el tablero."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    if rapido:
        flags += cv2.CALIB_CB_FAST_CHECK
    ok, esquinas = cv2.findChessboardCorners(gris, ESQUINAS, flags)
    if not ok:
        return None
    esquinas = cv2.cornerSubPix(gris, esquinas, (5, 5), (-1, -1), CRITERIO)
    ok, rvec, tvec = cv2.solvePnP(objp, esquinas, K, dist)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    c = R @ centro + tvec.ravel()                 # centro del tablero visto desde la camara (mm)
    Z = c[2]                                      # distancia a lo largo del eje de la camara
    distancia = float(np.linalg.norm(c))          # distancia en linea recta al centro
    normal = R[:, 2]                              # hacia donde "mira" el tablero
    inclinacion = float(np.degrees(np.arccos(abs(normal[2]))))  # 0 = tablero de frente
    return Z, distancia, inclinacion, rvec, tvec, esquinas


def main():
    K, dist, ancho, alto, sq = cargar_calibracion()
    objp, centro = puntos_tablero(sq)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    ok, frame = cap.read()
    if not ok:
        print("No se pudo leer la camara. Revisa CAM_INDEX.")
        return
    h, w = frame.shape[:2]
    if (w, h) != (ancho, alto):
        print(f"ERROR: la camara da {w} x {h} pero se calibro en {ancho} x {alto}. Las medidas saldrian mal.")
        return

    hist_z, hist_d = deque(maxlen=PROMEDIO), deque(maxlen=PROMEDIO)
    n = 0
    print("\nESPACIO = imprimir medicion | Q = salir\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Fallo la lectura de la camara")
            break
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        r = medir(gris, K, dist, objp, centro)

        if r is None:
            hist_z.clear()
            hist_d.clear()
            cv2.putText(frame, "No se ve el tablero completo", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            Z, d, incl, rvec, tvec, esquinas = r
            hist_z.append(Z)
            hist_d.append(d)
            cv2.drawChessboardCorners(frame, ESQUINAS, esquinas, True)
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 3 * sq)
            zm, dm = np.mean(hist_z), np.mean(hist_d)
            color = (0, 200, 0) if incl < 5 else (0, 165, 255)
            cv2.putText(frame, f"Z (de frente): {zm:6.1f} mm", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            cv2.putText(frame, f"Linea recta al centro: {dm:6.1f} mm", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Inclinacion: {incl:4.1f} grados" + ("  (OK, de frente)" if incl < 5 else ""),
                        (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, "ESPACIO = imprimir | Q = salir", (10, alto - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("medir distancia", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord(" ") and hist_z:
            n += 1
            print(f"Medicion {n}: Z = {np.mean(hist_z):.1f} mm | linea recta = {np.mean(hist_d):.1f} mm"
                  f" | inclinacion = {incl:.1f} grados")
        elif tecla == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
