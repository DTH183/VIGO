import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("controls")
names = ["H min", "H max", "S min", "S max", "V min", "V max"]
start = [78, 98, 80, 255, 80, 255]      # turquoise starting values
maxv = [179, 179, 255, 255, 255, 255]
for n, v, m in zip(names, start, maxv):
    cv2.createTrackbar(n, "controls", v, m, lambda x: None)

def g(n):
    return cv2.getTrackbarPos(n, "controls")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera read failed")
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = (g("H min"), g("S min"), g("V min"))
    high = (g("H max"), g("S max"), g("V max"))
    mask = cv2.inRange(hsv, low, high)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = False
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 800:
            found = True
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    label = "TURQUOISE: YES" if found else "TURQUOISE: no"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255) if found else (200, 200, 200), 2)
    cv2.imshow("camera", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("LOW  =", low)
        print("HIGH =", high)
        break

cap.release()
cv2.destroyAllWindows()
