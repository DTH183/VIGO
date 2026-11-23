"""
mano_ojo.py - Fase B: calibracion mano-ojo (camara montada en el gripper).

Este script NUNCA mueve el brazo: solo LEE su posicion.
El brazo se mueve a mano desde UFACTORY Studio (http://IP:18333), a velocidad baja.

Modos:
  python3 mano_ojo.py              -> capturar poses (ESPACIO = capturar, B = borrar ultima, Q = salir)
  python3 mano_ojo.py calcular     -> calcular la calibracion con las capturas guardadas
  python3 mano_ojo.py borrar 7     -> borrar la captura numero 7 (si salio mala)

Necesita calib_camara.npz (Fase A).
"""
import json
import os
import sys

import cv2
import numpy as np

# ---------- Ajustes ----------
ARM_IP = '192.168.0.150'   # IP del robot que se este usando
CAM_INDEX = 0
ESQUINAS = (9, 6)          # esquinas internas del tablero
SQUARE_MM = 15.0           # <- medida REAL del cuadro del tablero CHICO (vernier)
MIN_CAPTURAS = 12

BASE = os.path.dirname(os.path.abspath(__file__))
CALIB_CAM = os.path.join(BASE, "calib_camara.npz")
CARPETA = os.path.join(BASE, "mano_ojo")
DATOS = os.path.join(CARPETA, "capturas.json")
SALIDA = os.path.join(BASE, "calib_mano_ojo.npz")
CRITERIO = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

METODOS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


# ---------- Utilidades ----------
def rpy_a_matriz(roll, pitch, yaw):
    """Convencion de UFACTORY: R = Rz(yaw) * Ry(pitch) * Rx(roll), angulos en grados."""
    r, p, y = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matriz_a_rpy(R):
    pitch = np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])


def angulo_entre(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def homogenea(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def cargar_camara():
    if not os.path.exists(CALIB_CAM):
        print("Falta calib_camara.npz. Primero haz la Fase A (calibrar.py).")
        sys.exit(1)
    d = np.load(CALIB_CAM)
    w, h = (int(v) for v in d["image_size"])
    return d["camera_matrix"], d["dist_coeffs"], w, h


def puntos_tablero():
    objp = np.zeros((ESQUINAS[0] * ESQUINAS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:ESQUINAS[0], 0:ESQUINAS[1]].T.reshape(-1, 2) * SQUARE_MM
    return objp


def detectar(gris, rapido=False):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    if rapido:
        flags += cv2.CALIB_CB_FAST_CHECK
    ok, esq = cv2.findChessboardCorners(gris, ESQUINAS, flags)
    if not ok:
        return None
    esq = cv2.cornerSubPix(gris, esq, (5, 5), (-1, -1), CRITERIO)
    return ordenar(gris, esq)


def ordenar(gris, esq):
    """Evita que OpenCV tome como origen esquinas opuestas en distintas fotos (giro de 180 grados).
    El tablero tiene los cuadros de las esquinas NEGROS de un lado y BLANCOS del otro:
    siempre dejamos el origen del lado negro."""
    p = esq.reshape(-1, 2)
    nx, ny = ESQUINAS
    ult = nx * ny - 1

    def color_fuera(i, j):          # color del cuadro de la esquina, por fuera de la esquina i
        q = p[i] + 0.5 * (p[i] - p[j])
        x, y = int(round(q[0])), int(round(q[1]))
        if 0 <= x < gris.shape[1] and 0 <= y < gris.shape[0]:
            return int(gris[y, x])
        return None

    a = color_fuera(0, nx + 1)
    b = color_fuera(ult, ult - (nx + 1))
    if a is not None and b is not None and a > b:
        return esq[::-1].copy()
    return esq


def cargar_datos():
    if os.path.exists(DATOS):
        with open(DATOS) as f:
            return json.load(f)
    return {"tcp_offset": None, "square_mm": SQUARE_MM, "capturas": []}


def guardar_datos(datos):
    os.makedirs(CARPETA, exist_ok=True)
    with open(DATOS, "w") as f:
        json.dump(datos, f, indent=2)


# ---------- Modo 1: capturar ----------
def capturar():
    K, dist, w, h = cargar_camara()
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ARM_IP)
    if not arm.connected:
        print("No se pudo conectar al brazo en", ARM_IP)
        return
    tcp = [round(float(v), 3) for v in arm.tcp_offset]
    print("TCP offset del robot:", tcp)
    if any(abs(v) > 1e-3 for v in tcp):
        print("OJO: el TCP offset NO es cero. Esta bien, pero debe ser el MISMO siempre que uses esta calibracion.")

    datos = cargar_datos()
    if datos["capturas"] and datos["tcp_offset"] != tcp:
        print("ERROR: el TCP offset cambio respecto a las capturas guardadas. Borra la carpeta mano_ojo o regresa el TCP.")
        arm.disconnect()
        return
    if datos["capturas"] and datos.get("square_mm") != SQUARE_MM:
        print("ERROR: SQUARE_MM cambio respecto a las capturas guardadas.")
        arm.disconnect()
        return
    datos["tcp_offset"] = tcp
    datos["square_mm"] = SQUARE_MM

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, frame = cap.read()
    if not ok or frame.shape[1] != w or frame.shape[0] != h:
        print(f"La camara no da {w} x {h} (la resolucion de la calibracion).")
        arm.disconnect()
        return

    print("\nMueve el brazo con UFACTORY Studio (velocidad baja). Cuando se detenga:")
    print("ESPACIO = capturar | B = borrar la ultima | Q = salir\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Fallo la camara")
                break
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            esq = detectar(gris, rapido=True)
            if esq is not None:
                cv2.drawChessboardCorners(frame, ESQUINAS, esq, True)
                cv2.putText(frame, "TABLERO OK", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            else:
                cv2.putText(frame, "No se ve el tablero completo", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            n = len(datos["capturas"])
            cv2.putText(frame, f"Capturas: {n} (min {MIN_CAPTURAS}) | ESPACIO B Q", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("mano-ojo", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord(" "):
                if arm.get_is_moving():
                    print("El brazo se esta moviendo. Espera a que se detenga.")
                    continue
                for _ in range(5):          # tirar cuadros viejos del buffer
                    cap.read()
                ok, limpia = cap.read()
                esq = detectar(cv2.cvtColor(limpia, cv2.COLOR_BGR2GRAY)) if ok else None
                if esq is None:
                    print("No capturada: no se ve el tablero completo.")
                    continue
                code, pose = arm.get_position(is_radian=False)
                if code != 0:
                    print("No capturada: error leyendo la posicion del brazo, code", code)
                    continue
                num = 1 + max([c["num"] for c in datos["capturas"]], default=0)
                nombre = f"pose_{num:02d}.png"
                os.makedirs(CARPETA, exist_ok=True)
                cv2.imwrite(os.path.join(CARPETA, nombre), limpia)
                datos["capturas"].append({"num": num, "imagen": nombre,
                                          "pose": [round(float(v), 4) for v in pose]})
                guardar_datos(datos)
                R_nueva = rpy_a_matriz(*pose[3:])
                giro = max([angulo_entre(R_nueva, rpy_a_matriz(*c["pose"][3:]))
                            for c in datos["capturas"][:-1]], default=0.0)
                print(f"Captura {num}: pose = {[round(v, 1) for v in pose]} | "
                      f"giro maximo vs. otras: {giro:.1f} grados")
            elif tecla == ord("b") and datos["capturas"]:
                c = datos["capturas"].pop()
                ruta = os.path.join(CARPETA, c["imagen"])
                if os.path.exists(ruta):
                    os.remove(ruta)
                guardar_datos(datos)
                print("Borrada la captura", c["num"])
            elif tecla == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arm.disconnect()


# ---------- Modo 2: calcular ----------
def calcular_desde(poses, rvecs, tvecs):
    """poses: lista [x,y,z,roll,pitch,yaw] (mm, grados). rvecs/tvecs: tablero visto por la camara."""
    R_g2b = [rpy_a_matriz(*p[3:]) for p in poses]
    t_g2b = [np.array(p[:3], float).reshape(3, 1) for p in poses]
    R_t2c = [cv2.Rodrigues(np.asarray(r, float))[0] for r in rvecs]
    t_t2c = [np.asarray(t, float).reshape(3, 1) for t in tvecs]

    resultados = []
    for nombre, metodo in METODOS.items():
        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=metodo)
        except cv2.error:
            continue
        X = homogenea(R_c2g, t_c2g)
        origenes = []
        for i in range(len(poses)):
            T = homogenea(R_g2b[i], t_g2b[i]) @ X @ homogenea(R_t2c[i], t_t2c[i])
            origenes.append(T[:3, 3])
        origenes = np.array(origenes)
        media = origenes.mean(axis=0)
        errores = np.linalg.norm(origenes - media, axis=1)
        dispersion = float(np.sqrt(np.mean(errores ** 2)))
        resultados.append((dispersion, nombre, R_c2g, t_c2g, media, errores))
    resultados.sort(key=lambda r: r[0])
    return resultados, R_g2b


def calcular():
    K, dist, w, h = cargar_camara()
    datos = cargar_datos()
    caps = datos["capturas"]
    if len(caps) < MIN_CAPTURAS:
        print(f"Hay {len(caps)} capturas; se necesitan al menos {MIN_CAPTURAS}.")
        return
    if datos.get("square_mm") != SQUARE_MM:
        print("OJO: SQUARE_MM no coincide con el de las capturas. Revisa.")
    objp = puntos_tablero()

    poses, rvecs, tvecs, nums = [], [], [], []
    for c in caps:
        img = cv2.imread(os.path.join(CARPETA, c["imagen"]))
        esq = detectar(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) if img is not None else None
        if esq is None:
            print("Ignorada (no se detecto el tablero):", c["imagen"])
            continue
        ok, rv, tv = cv2.solvePnP(objp, esq, K, dist)
        if ok:
            poses.append(c["pose"])
            rvecs.append(rv)
            tvecs.append(tv)
            nums.append(c["num"])

    resultados, R_g2b = calcular_desde(poses, rvecs, tvecs)
    if not resultados:
        print("No se pudo calcular. Toma capturas con mas variedad de giros.")
        return

    giro = max(angulo_entre(R_g2b[i], R_g2b[j]) for i in range(len(R_g2b)) for j in range(i + 1, len(R_g2b)))

    print("\n========== RESULTADO MANO-OJO ==========")
    print(f"Capturas usadas: {len(poses)} | giro maximo entre poses: {giro:.1f} grados")
    if giro < 20:
        print("  -> OJO: muy poco giro. Agrega poses mas inclinadas (20-30 grados).")
    print("\nDispersion por metodo (menor = mejor):")
    for disp, nombre, *_ in resultados:
        print(f"  {nombre:11s} {disp:6.2f} mm")

    disp, nombre, R_c2g, t_c2g, media, errores = resultados[0]
    print(f"\nMejor metodo: {nombre}")
    print(f"Dispersion del tablero: {disp:.2f} mm", end="  ")
    if disp < 3:
        print("-> BUENA")
    elif disp < 6:
        print("-> ACEPTABLE (se puede mejorar)")
    else:
        print("-> MALA: revisa las peores capturas o toma mas con mas giro")

    t = t_c2g.ravel()
    print(f"\nCamara respecto al TCP: x = {t[0]:.1f}  y = {t[1]:.1f}  z = {t[2]:.1f} mm "
          f"(distancia {np.linalg.norm(t):.1f} mm)")
    print("Giro de la camara respecto al TCP (roll, pitch, yaw):", np.round(matriz_a_rpy(R_c2g), 1))
    print(f"\nEsquina de origen del tablero en coordenadas del robot: "
          f"x = {media[0]:.1f}  y = {media[1]:.1f}  z = {media[2]:.1f} mm")
    print("  (esa z es la altura de la SUPERFICIE del tablero = Z_MESA + grosor del tablero)")

    peores = sorted(zip(errores, nums), reverse=True)[:3]
    print("\nCapturas con mas error (candidatas a borrar):")
    for e, n in peores:
        print(f"  captura {n}: {e:.2f} mm   -> python3 mano_ojo.py borrar {n}")

    np.savez(SALIDA, R_cam2tcp=R_c2g, t_cam2tcp=t_c2g, metodo=nombre, dispersion_mm=disp,
             origen_tablero_base=media, tcp_offset=np.array(datos["tcp_offset"]), n_capturas=len(poses))
    print("\nGuardado en:", SALIDA)


def borrar(num):
    datos = cargar_datos()
    antes = len(datos["capturas"])
    resto = []
    for c in datos["capturas"]:
        if c["num"] == num:
            ruta = os.path.join(CARPETA, c["imagen"])
            if os.path.exists(ruta):
                os.remove(ruta)
        else:
            resto.append(c)
    datos["capturas"] = resto
    guardar_datos(datos)
    print("Borrada" if len(resto) < antes else "No existe esa captura", num)


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "capturar"
    if modo == "calcular":
        calcular()
    elif modo == "borrar" and len(sys.argv) > 2:
        borrar(int(sys.argv[2]))
    else:
        capturar()
