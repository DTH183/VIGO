"""
validar.py - Fase D: comprobar que la camara da coordenadas correctas del robot.

  python3 validar.py          -> SIMULACION: calcula y muestra el plan, el brazo NO se mueve
  python3 validar.py real     -> mueve el brazo a ALTURA_TCP mm arriba de la esquina ORIGEN del tablero

Teclas (clic en la ventana "validar"): ESPACIO = calcular objetivo | Q = salir
Usa calib_camara.npz, calib_mano_ojo.npz y los ajustes de mano_ojo.py (IP, tablero).

SEGURIDAD: supervisor presente, mano en el paro de emergencia, area despejada.
"""
import os
import sys

import termios
import time

import cv2
import numpy as np

import mano_ojo as mo   # reutiliza IP, tablero, deteccion y conversiones

REAL = "real" in sys.argv

# ---------- Ajustes de seguridad ----------
ALTURA_TCP = 200.0     # mm: la BRIDA (TCP con offset 0) queda 200 mm arriba de la esquina
SPEED = 20             # mm/s (lento)
ACC = 100              # mm/s^2 (suave)
R_MAX = 400.0          # mm: distancia horizontal maxima desde la base (alcance Lite 6 = 440)
VIAJE_MAX = 350.0      # mm: distancia maxima entre la pose actual y el objetivo
Z_MAX = 600.0          # mm: altura maxima permitida para la brida

SALIDA_HE = os.path.join(mo.BASE, "calib_mano_ojo.npz")


def preguntar(msg):
    """Borra lo que se haya tecleado antes en la terminal y luego pregunta.
    Asi un 'SI' escrito antes de tiempo NO confirma un movimiento despues."""
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except (termios.error, ValueError, OSError):
        pass
    return input(msg)


def preparar(arm):
    """Deja el brazo listo para moverse JUSTO antes de cada movimiento.
    (Si en medio se uso Studio, el robot puede quedar en otro modo/estado -> code 9.)"""
    if arm.error_code != 0:
        print("El brazo tiene un ERROR activo (code", arm.error_code, "). Resuelvelo en Studio; NO se mueve.")
        return False
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)
    print(f"Estado del brazo: state = {arm.state}, mode = {arm.mode}, error = {arm.error_code}")
    if arm.mode != 0 or arm.state not in (1, 2):
        print("El brazo NO quedo listo (debe ser mode 0 y state 2). Revisa Studio: Manual Mode apagado, sin errores.")
        return False
    return True


# ---------- Calculos (sin robot) ----------
def cargar_mano_ojo():
    if not os.path.exists(SALIDA_HE):
        print("Falta calib_mano_ojo.npz. Primero haz la Fase B (mano_ojo.py).")
        sys.exit(1)
    d = np.load(SALIDA_HE)
    return mo.homogenea(d["R_cam2tcp"], d["t_cam2tcp"]), [float(v) for v in d["tcp_offset"]]


def origen_en_base(pose, rvec, tvec, X):
    """Posicion (mm) de la esquina ORIGEN del tablero en coordenadas del robot."""
    T_b_g = mo.homogenea(mo.rpy_a_matriz(*pose[3:]), pose[:3])
    T_c_t = mo.homogenea(cv2.Rodrigues(rvec)[0], tvec)
    return (T_b_g @ X @ T_c_t)[:3, 3]


def plan(actual, punto):
    """Lista de poses: subir si hace falta -> moverse horizontal -> bajar a ALTURA_TCP sobre el punto.
    Herramienta apuntando hacia abajo (roll 180, pitch 0) y se conserva el yaw actual."""
    x0, y0, z0, r0, p0, yaw0 = [float(v) for v in actual]
    xt, yt, zt = float(punto[0]), float(punto[1]), float(punto[2]) + ALTURA_TCP
    z_seg = max(z0, zt)
    pasos = []
    if z0 < z_seg:
        pasos.append(("subir", [x0, y0, z_seg, r0, p0, yaw0]))
    pasos.append(("mover encima", [xt, yt, z_seg, 180.0, 0.0, yaw0]))
    if z_seg > zt:
        pasos.append(("bajar", [xt, yt, zt, 180.0, 0.0, yaw0]))
    return pasos


def revisar_limites(actual, pasos):
    problemas = []
    for nombre, p in pasos:
        if np.hypot(p[0], p[1]) > R_MAX:
            problemas.append(f"'{nombre}' queda a {np.hypot(p[0], p[1]):.0f} mm de la base (max {R_MAX:.0f})")
        if p[2] > Z_MAX:
            problemas.append(f"'{nombre}' queda en z = {p[2]:.0f} mm (max {Z_MAX:.0f})")
    destino = pasos[-1][1]
    viaje = np.linalg.norm(np.array(destino[:3]) - np.array(actual[:3]))
    if viaje > VIAJE_MAX:
        problemas.append(f"el viaje es de {viaje:.0f} mm (max {VIAJE_MAX:.0f}); acerca el brazo primero")
    return problemas


# ---------- Programa ----------
def main():
    K, dist, w, h = mo.cargar_camara()
    X, tcp_calib = cargar_mano_ojo()
    objp = mo.puntos_tablero()

    from xarm.wrapper import XArmAPI
    arm = XArmAPI(mo.ARM_IP)
    if not arm.connected:
        print("No se pudo conectar al brazo en", mo.ARM_IP)
        return
    tcp = [float(v) for v in arm.tcp_offset]
    if not np.allclose(tcp, tcp_calib, atol=1e-2):
        print("ERROR: el TCP offset del robot", tcp, "no es el de la calibracion", tcp_calib)
        arm.disconnect()
        return

    if REAL:
        arm.clean_warn()
        arm.clean_error()
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
        if arm.error_code != 0:
            print("El brazo tiene un error, NO se va a mover. Code:", arm.error_code)
            arm.disconnect()
            return
        print("MODO REAL: el brazo SI se movera (siempre se pide confirmacion).")
    else:
        print("SIMULACION: el brazo NO se mueve. Usa 'python3 validar.py real' para moverlo.")

    cap = cv2.VideoCapture(mo.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    print("\nESPACIO = calcular objetivo | Q = salir\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Fallo la camara")
                break
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            esq = mo.detectar(gris, rapido=True)
            if esq is not None:
                cv2.drawChessboardCorners(frame, mo.ESQUINAS, esq, True)
                o = tuple(int(v) for v in esq[0].ravel())
                cv2.circle(frame, o, 12, (255, 0, 255), 3)
                cv2.putText(frame, "ORIGEN", (o[0] + 14, o[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                ok_pnp, rv, tv = cv2.solvePnP(objp, esq, K, dist)
                if ok_pnp:
                    p = origen_en_base([float(v) for v in arm.position], rv, tv, X)
                    cv2.putText(frame, f"Origen en robot: x {p[0]:.0f}  y {p[1]:.0f}  z {p[2]:.0f} mm",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
            else:
                cv2.putText(frame, "No se ve el tablero completo", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, ("REAL" if REAL else "SIMULACION") + " | ESPACIO calcular | Q salir",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("validar", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla != ord(" "):
                continue

            # ----- calcular con datos frescos -----
            if arm.get_is_moving():
                print("El brazo se esta moviendo; espera.")
                continue
            for _ in range(5):
                cap.read()
            ok, limpia = cap.read()
            esq = mo.detectar(cv2.cvtColor(limpia, cv2.COLOR_BGR2GRAY)) if ok else None
            if esq is None:
                print("No se ve el tablero completo.")
                continue
            code, actual = arm.get_position(is_radian=False)
            if code != 0:
                print("Error leyendo la posicion, code", code)
                continue
            actual = [float(v) for v in actual]
            _, rv, tv = cv2.solvePnP(objp, esq, K, dist)
            punto = origen_en_base(actual, rv, tv, X)
            pasos = plan(actual, punto)

            print("\n----------------------------------------------")
            print(f"Esquina ORIGEN en coordenadas del robot: x = {punto[0]:.1f}  y = {punto[1]:.1f}  z = {punto[2]:.1f} mm")
            print("Plan (x, y, z, roll, pitch, yaw):")
            for nombre, p in pasos:
                print(f"  {nombre:13s} {[round(v, 1) for v in p]}")
            print(f"La BRIDA quedara {ALTURA_TCP:.0f} mm arriba de la esquina.")
            problemas = revisar_limites(actual, pasos)
            if problemas:
                print("NO SE MUEVE por seguridad:")
                for pr in problemas:
                    print("  -", pr)
                continue
            if not REAL:
                print("(Simulacion: no se movio nada)")
                continue

            # ----- mover (solo en modo real, con confirmacion) -----
            cv2.waitKey(1)
            print("\n>>> CONFIRMACION: escribe aqui, en esta TERMINAL <<<")
            resp = preguntar("Mano en el paro de emergencia. Escribe SI y Enter para mover (otra cosa cancela): ")
            if resp.strip().upper() != "SI":
                print("Cancelado.")
                continue
            if not preparar(arm):
                continue
            exito = True
            for nombre, p in pasos:
                code = arm.set_position(*p, speed=SPEED, mvacc=ACC, wait=True)
                print(f"{nombre}: code {code}")
                if code != 0:
                    print("Movimiento fallido; se detiene. Estado:", arm.state, "| error:", arm.error_code)
                    exito = False
                    break
            if exito:
                print("\nLISTO. Revisa con una regla:")
                print("  1) la punta del gripper debe quedar justo encima de la esquina ORIGEN")
                print(f"  2) altura punta -> tablero = {ALTURA_TCP:.0f} mm - (distancia brida -> punta)")
            preguntar("Enter para regresar a la pose donde estaba...")
            if preparar(arm):
                z_seg = max(arm.position[2], actual[2])
                regreso_ok = True
                for p in ([*arm.position[:2], z_seg, *arm.position[3:]], [*actual[:2], z_seg, *actual[3:]], actual):
                    code = arm.set_position(*[float(v) for v in p], speed=SPEED, mvacc=ACC, wait=True)
                    if code != 0:
                        print("Fallo al regresar, code", code)
                        regreso_ok = False
                        break
                if regreso_ok:
                    print("De regreso.")
            for _ in range(5):
                cap.read()
    except KeyboardInterrupt:
        print("Interrumpido por el usuario")
        if REAL:
            arm.set_state(4)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arm.disconnect()


if __name__ == "__main__":
    main()
