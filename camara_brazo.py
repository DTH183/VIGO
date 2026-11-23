import sys
import cv2

ARM_IP = '192.168.0.150'
SIMULATE = 'real' not in sys.argv     # default: arm does NOT move

LOW = (78, 80, 80)       # <- paste your values from test_camara.py
HIGH = (98, 255, 255)
MIN_AREA = 800           # ignore small blobs (noise)
CONFIRM = 5              # frames in a row before acting

STEP = 20     # mm
SPEED = 20    # mm/s (slow)
ACC = 100     # mm/s^2 (gentle)

# ---------- Arm setup ----------
arm = None
if not SIMULATE:
    from xarm.wrapper import XArmAPI
    arm = XArmAPI(ARM_IP)
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    if arm.error_code != 0:
        print("Arm error, NOT moving. Code:", arm.error_code)
        arm.disconnect()
        raise SystemExit
    print("Start position:", arm.get_position())
else:
    print("SIMULATION: the arm will not move. Run with 'real' to move it.")

# ---------- Functions ----------
def turquoise_detected(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOW, HIGH)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    return cv2.contourArea(max(cnts, key=cv2.contourArea)) > MIN_AREA

def move_z(dz):
    if arm is None:
        print(f"[SIMULATED] move z {dz:+d} mm")
        return 0
    code = arm.set_position(z=dz, relative=True, speed=SPEED, mvacc=ACC, wait=True)
    print(f"move z {dz:+d} mm -> code {code}")
    return code

# ---------- Main loop ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

raised = False
seen_n = 0
absent_n = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break

        found = turquoise_detected(frame)
        seen_n = seen_n + 1 if found else 0
        absent_n = 0 if found else absent_n + 1

        if seen_n >= CONFIRM and not raised:
            if move_z(STEP) != 0:
                break
            raised = True
            for _ in range(5):
                cap.read()      # flush old frames after the move
        elif absent_n >= CONFIRM and raised:
            if move_z(-STEP) != 0:
                break
            raised = False
            for _ in range(5):
                cap.read()

        state = "UP" if raised else "DOWN"
        cv2.putText(frame, f"{'TURQUOISE' if found else 'no object'} | arm: {state} | q = quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("camera + arm", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            if raised:
                move_z(-STEP)   # return to start height before exiting
                raised = False
            break
except KeyboardInterrupt:
    print("Interrupted")
finally:
    cap.release()
    cv2.destroyAllWindows()
    if arm is not None:
        print("Final position:", arm.get_position())
        arm.disconnect()
