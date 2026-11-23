import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Ventana con barras para ajustar el color en vivo
cv2.namedWindow("controles")
for nombre, valor, maximo in [("H min", 35, 179), ("H max", 85, 179),
                              ("S min", 80, 255), ("S max", 255, 255),
                              ("V min", 80, 255), ("V max", 255, 255)]:
    cv2.createTrackbar(nombre, "controles", valor, maximo, lambda x: None)

while True:
    ok, frame = cap.read()
    if not ok:
        print("No se pudo leer la cámara")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = tuple(cv2.getTrackbarPos(n, "controles") for n in ("H min", "S min", "V min"))
    high = tuple(cv2.getTrackbarPos(n, "controles") for n in ("H max", "S max", "V max"))
    mask = cv2.inRange(hsv, low, high)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            M = cv2.moments(c)
            u, v = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.circle(frame, (u, v), 8, (0, 0, 255), -1)
            print("Objeto en pixel:", u, v)

    cv2.imshow("camara", frame)
    cv2.imshow("mascara", mask)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
