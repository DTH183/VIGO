"""
cubo.py - La camara encuentra el cubo VERDE (30 mm) y el robot pone el gripper ENCIMA de el.

Modos:
  python3 cubo.py color   -> ajustar el color verde con barras (S = guardar, Q = salir)
  python3 cubo.py         -> SIMULACION: detecta y muestra el plan; el brazo NO se mueve
  python3 cubo.py real    -> mueve el brazo (pide SI antes de cada movimiento)

Secuencia en modo real (ESPACIO):
  1) detecta el cubo y calcula su posicion
  2) pone la CAMARA encima del cubo y lo vuelve a detectar (mas preciso)
  3) pone la PUNTA del gripper HOLGURA mm arriba de la cara superior del cubo
Usa calib_camara.npz, calib_mano_ojo.npz, mano_ojo.py y validar.py (misma carpeta).

SEGURIDAD: supervisor presente, mano en el paro de emergencia, area despejada.
"""
import json
import os
import sys

import cv2
import numpy as np

import mano_ojo as mo
import validar as va    # preparar(), preguntar(), revisar_limites(), limites de seguridad

REAL = "real" in sys.argv

# ---------- Ajustes ----------
LADO_CUBO = 30.0        # mm
Z_MESA = -24.0          # mm: altura de la mesa en coordenadas del robot  <- AJUSTAR (ver instrucciones)
LARGO_GRIPPER = 120.0   # mm: de la brida (TCP original) a la punta del gripper
HOLGURA = 0.0           # mm: la punta queda esta distancia ARRIBA de la cara superior del cubo
                        #     (antes 50; ahora baja 50 mm mas: punta a la altura de la cara superior)
OFFSET_X = 10.0         # mm: correccion en +x del robot (centro del gripper vs. camara)
YAW_REF = -91.2         # grados: yaw con el que se midio OFFSET_X (si cambia el yaw, se gira solo)
TOLERANCIA = 5.0        # mm: el lado medido debe estar entre 25 y 35 mm
Z_PUNTA_MIN = 10.0      # mm: la punta NUNCA baja a menos de esto sobre la mesa
SPEED_FINAL = 10        # mm/s: velocidad de la ultima bajada hacia el cubo
ALTURA_CAMARA = 250.0   # mm: altura de la camara sobre el cubo en el paso 2
MIN_AREA = 300          # px: manchas mas chicas se ignoran
SPEED, ACC = va.SPEED, va.ACC

COLOR_JSON = os.path.join(mo.BASE, "color_cubo.json")
LOW_DEF, HIGH_DEF = (45, 100, 70), (80, 255, 255)   # verde (ajustar con el modo color)

Z_TOP = Z_MESA + LADO_CUBO   # cara superior del cubo


# ---------- Color ----------
def cargar_color():
    if os.path.exists(COLOR_JSON):
        with open(COLOR_JSON) as f:
            d = json.load(f)
        return tuple(d["low"]), tuple(d["high"])
    return LOW_DEF, HIGH_DEF


def modo_color():
    K, dist, w, h = mo.cargar_camara()
    low, high = cargar_color()
    cap = cv2.VideoCapture(mo.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cv2.namedWindow("controles")
    for n, v, m in zip(["H min", "S min", "V min", "H max", "S max", "V max"], [*low, *high],
                       [179, 255, 255, 179, 255, 255]):
        cv2.createTrackbar(n, "controles", int(v), m, lambda x: None)
    print("Ajusta hasta que en 'mascara' SOLO el cubo salga blanco. S = guardar | Q = salir")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = lambda n: cv2.getTrackbarPos(n, "controles")
        low = (g("H min"), g("S min"), g("V min"))
        high = (g("H max"), g("S max"), g("V max"))
        r = detectar_cubo(frame, low, high)
        if r is not None:
            cv2.drawContours(frame, [r[1].astype(int)], 0, (0, 0, 255), 2)
        cv2.imshow("camara", frame)
        cv2.imshow("mascara", mascara(frame, low, high))
        t = cv2.waitKey(1) & 0xFF
        if t == ord("s"):
            with open(COLOR_JSON, "w") as f:
                json.dump({"low": low, "high": high}, f)
            print("Guardado:", low, high)
        elif t == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# ---------- Vision ----------
def mascara(frame, low, high):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)    # quita puntitos
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)  # tapa huecos


def detectar_cubo(frame, low, high):
    """Devuelve (centro_px, caja_4x2) de la mancha verde mas grande, o None."""
    m = mascara(frame, low, high)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None
    M = cv2.moments(c)
    centro = (M["m10"] / M["m00"], M["m01"] / M["m00"])
    caja = cv2.boxPoints(cv2.minAreaRect(c))
    return centro, caja


def camara_en_base(pose, X):
    return mo.homogenea(mo.rpy_a_matriz(*pose[3:]), pose[:3]) @ X


def pixel_a_plano(u, v, K, dist, T_b_c, z_plano):
    """Rayo que sale de la camara por el pixel (u, v) -> punto donde choca con el plano z = z_plano."""
    n = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, dist).ravel()
    d = T_b_c[:3, :3] @ np.array([n[0], n[1], 1.0])
    o = T_b_c[:3, 3]
    if abs(d[2]) < 1e-6:
        return None
    s = (z_plano - o[2]) / d[2]
    return o + s * d if s > 0 else None


def medir_cubo(det, pose, K, dist, X):
    """Posicion (centro de la cara superior) y tamano aparente del cubo, en mm del robot."""
    T_b_c = camara_en_base(pose, X)
    p = pixel_a_plano(*det[0], K, dist, T_b_c, Z_TOP)
    esquinas = [pixel_a_plano(u, v, K, dist, T_b_c, Z_TOP) for u, v in det[1]]
    if p is None or any(e is None for e in esquinas):
        return None, None, None
    lados = [np.linalg.norm(esquinas[i][:2] - esquinas[(i + 1) % 4][:2]) for i in range(4)]
    e0, e1 = esquinas[0], esquinas[1]
    angulo = float(np.degrees(np.arctan2(e1[1] - e0[1], e1[0] - e0[0]))) % 90
    return p, float(np.mean(lados)), angulo


def tamano_ok(lado):
    return LADO_CUBO - TOLERANCIA <= lado <= LADO_CUBO + TOLERANCIA


# ---------- Planes de movimiento ----------
def pose_camara_encima(p, yaw, X):
    """Brida tal que la CAMARA queda ALTURA_CAMARA mm arriba de p, viendo hacia abajo."""
    R = mo.rpy_a_matriz(180.0, 0.0, yaw)
    off = R @ X[:3, 3]
    return [float(p[0] - off[0]), float(p[1] - off[1]), float(p[2] + ALTURA_CAMARA - off[2]), 180.0, 0.0, yaw]


def offset_gripper(yaw):
    """OFFSET_X se midio en +x del robot con YAW_REF. Como el desfase va 'pegado' al gripper,
    si el yaw es otro se gira la correccion con el."""
    a = np.radians(yaw - YAW_REF)
    return OFFSET_X * np.cos(a), OFFSET_X * np.sin(a)


def pose_gripper_encima(p, yaw):
    """Brida tal que la PUNTA del gripper queda HOLGURA mm arriba de la cara superior (p),
    corregida por el desfase del gripper (OFFSET_X)."""
    dx, dy = offset_gripper(yaw)
    return [float(p[0] + dx), float(p[1] + dy), float(p[2] + LARGO_GRIPPER + HOLGURA), 180.0, 0.0, yaw]


def plan_a(actual, destino):
    """Moverse en horizontal a la altura mas alta y luego subir/bajar. Nunca baja en diagonal."""
    z_seg = max(actual[2], destino[2])
    pasos = []
    if actual[2] < z_seg:
        pasos.append(("subir", [*actual[:2], z_seg, *actual[3:]]))
    pasos.append(("mover encima", [*destino[:2], z_seg, *destino[3:]]))
    if z_seg > destino[2]:
        pasos.append(("bajar", list(destino)))
    return pasos


def seguro(actual, pasos):
    probs = va.revisar_limites(actual, pasos)
    z_min = Z_MESA + Z_PUNTA_MIN + LARGO_GRIPPER   # la punta nunca a menos de Z_PUNTA_MIN de la mesa
    for nombre, p in pasos:
        if p[2] < z_min:
            probs.append(f"'{nombre}' baja la brida a z = {p[2]:.0f} (minimo {z_min:.0f})")
    return probs


def ejecutar(arm, pasos, lento_al_final=False):
    if not va.preparar(arm):
        return False
    for i, (nombre, p) in enumerate(pasos):
        vel = SPEED_FINAL if (lento_al_final and i == len(pasos) - 1) else SPEED
        code = arm.set_position(*p, speed=vel, mvacc=ACC, wait=True)
        print(f"  {nombre}: code {code}")
        if code != 0:
            print("  Movimiento fallido; se detiene. Estado:", arm.state, "| error:", arm.error_code)
            return False
    return True


def confirmar(texto):
    print(f"\n>>> {texto}")
    print(">>> CONFIRMACION: escribe aqui, en esta TERMINAL <<<")
    return va.preguntar("Mano en el paro de emergencia. Escribe SI y Enter (otra cosa cancela): ").strip().upper() == "SI"


def imprimir_plan(pasos):
    for nombre, p in pasos:
        print(f"  {nombre:13s} {[round(v, 1) for v in p]}")


def foto_fresca(cap, n=5):
    for _ in range(n):
        cap.read()
    return cap.read()


# ---------- Programa principal ----------
def main():
    K, dist, w, h = mo.cargar_camara()
    X, tcp_calib = va.cargar_mano_ojo()
    low, high = cargar_color()
    print("Color:", low, high, "| Z_MESA =", Z_MESA, "| cara superior en z =", Z_TOP)

    from xarm.wrapper import XArmAPI
    arm = XArmAPI(mo.ARM_IP)
    if not arm.connected:
        print("No se pudo conectar al brazo en", mo.ARM_IP)
        return
    if not np.allclose([float(v) for v in arm.tcp_offset], tcp_calib, atol=1e-2):
        print("ERROR: el TCP offset cambio respecto a la calibracion. Debe ser", tcp_calib)
        arm.disconnect()
        return
    print("MODO REAL: el brazo SI se movera (pide SI antes de cada movimiento)." if REAL else
          "SIMULACION: el brazo NO se mueve. Usa 'python3 cubo.py real' para moverlo.")

    cap = cv2.VideoCapture(mo.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    print("\nESPACIO = buscar el cubo y moverse | Q = salir\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Fallo la camara")
                break
            det = detectar_cubo(frame, low, high)
            if det is not None:
                p, lado, ang = medir_cubo(det, [float(v) for v in arm.position], K, dist, X)
                cv2.drawContours(frame, [det[1].astype(int)], 0, (0, 0, 255), 2)
                cv2.circle(frame, tuple(int(v) for v in det[0]), 5, (0, 0, 255), -1)
                if p is not None:
                    color = (0, 200, 0) if tamano_ok(lado) else (0, 0, 255)
                    cv2.putText(frame, f"Cubo: x {p[0]:.0f}  y {p[1]:.0f}  z {p[2]:.0f} mm", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, f"Lado medido: {lado:.0f} mm (real {LADO_CUBO:.0f})", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(frame, "No se ve el cubo verde", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, ("REAL" if REAL else "SIMULACION") + " | ESPACIO buscar | Q salir",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("cubo", frame)

            t = cv2.waitKey(1) & 0xFF
            if t == ord("q"):
                break
            if t != ord(" "):
                continue

            # ===== 1) Detectar desde donde esta =====
            if arm.get_is_moving():
                print("El brazo se esta moviendo; espera.")
                continue
            ok, img = foto_fresca(cap)
            det = detectar_cubo(img, low, high) if ok else None
            code, inicio = arm.get_position(is_radian=False)
            if det is None or code != 0:
                print("No se ve el cubo (o no se pudo leer el brazo).")
                continue
            inicio = [float(v) for v in inicio]
            p1, lado, ang = medir_cubo(det, inicio, K, dist, X)
            print("\n==============================================")
            if p1 is None:
                print("No se pudo calcular la posicion (la camara no ve hacia la mesa).")
                continue
            print(f"1) Cubo: x = {p1[0]:.1f}  y = {p1[1]:.1f}  (cara superior z = {Z_TOP:.1f})"
                  f" | lado medido {lado:.1f} mm | giro {ang:.0f} grados")
            if not tamano_ok(lado):
                print("   El tamano no parece el del cubo: revisa el color (modo 'color') o Z_MESA. No se mueve.")
                continue
            yaw = inicio[5]

            # ===== 2) Poner la camara encima y volver a medir =====
            pasos = plan_a(inicio, pose_camara_encima(p1, yaw, X))
            print("2) Plan para poner la CAMARA encima del cubo:")
            imprimir_plan(pasos)
            probs = seguro(inicio, pasos)
            if probs:
                print("NO SE MUEVE por seguridad:", *probs, sep="\n  - ")
                continue
            if not REAL:
                pasos_g = plan_a(pasos[-1][1], pose_gripper_encima(p1, yaw))
                print("3) Plan para poner el GRIPPER encima (con la medicion 1):")
                imprimir_plan(pasos_g)
                print("(Simulacion: no se movio nada)")
                continue
            if not confirmar("Paso 2: poner la CAMARA encima del cubo"):
                print("Cancelado.")
                continue
            if not ejecutar(arm, pasos):
                continue
            ok, img = foto_fresca(cap)
            det = detectar_cubo(img, low, high) if ok else None
            code, pose2 = arm.get_position(is_radian=False)
            p2, lado2 = None, None
            if det is not None and code == 0:
                p2, lado2, _ = medir_cubo(det, [float(v) for v in pose2], K, dist, X)
            if p2 is None:
                print("Desde arriba ya no se ve el cubo. No se sigue.")
            elif not tamano_ok(lado2):
                print(f"   Medicion 2: el lado mide {lado2:.1f} mm (fuera de {LADO_CUBO:.0f} +- {TOLERANCIA:.0f}). No se baja.")
            else:
                print(f"   Medicion 2: x = {p2[0]:.1f}  y = {p2[1]:.1f} | lado medido {lado2:.1f} mm "
                      f"(cambio vs. medicion 1: {np.linalg.norm(p2[:2] - p1[:2]):.1f} mm)")

                # ===== 3) Poner el gripper encima =====
                actual = [float(v) for v in arm.position]
                pasos = plan_a(actual, pose_gripper_encima(p2, yaw))
                print("3) Plan para poner la PUNTA del gripper encima del cubo:")
                imprimir_plan(pasos)
                probs = seguro(actual, pasos)
                if probs:
                    print("NO SE MUEVE por seguridad:", *probs, sep="\n  - ")
                elif confirmar(f"Paso 3: punta del gripper a {HOLGURA:.0f} mm de la cara superior del cubo "
                               f"(+{OFFSET_X:.0f} mm en x). Gripper ABIERTO."):
                    if ejecutar(arm, pasos, lento_al_final=True):
                        print(f"\nLISTO. Revisa: el gripper debe quedar centrado sobre el cubo y la punta a ~{HOLGURA:.0f} mm de su cara superior.")
                else:
                    print("Cancelado.")

            # ===== Regresar =====
            va.preguntar("\nEnter para regresar a la pose inicial...")
            actual = [float(v) for v in arm.position]
            ejecutar(arm, plan_a(actual, inicio))
            foto_fresca(cap)
    except KeyboardInterrupt:
        print("Interrumpido por el usuario")
        if REAL:
            arm.set_state(4)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arm.disconnect()


if __name__ == "__main__":
    if "color" in sys.argv:
        modo_color()
    else:
        main()
