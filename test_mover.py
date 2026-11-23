from xarm.wrapper import XArmAPI

arm = XArmAPI('192.168.0.150')

SPEED = 20    # mm/s (slow)
ACC = 100     # mm/s^2 (gentle)
STEP = 20     # mm

# Pre-flight
arm.clean_warn()
arm.clean_error()
arm.motion_enable(True)
arm.set_mode(0)
arm.set_state(0)

if arm.error_code != 0:
    print("Arm has an error, NOT moving. Code:", arm.error_code)
    arm.disconnect()
    raise SystemExit

print("Start position:", arm.get_position())

try:
    input("Hand on the e-stop. Press Enter to move UP 20 mm...")
    code = arm.set_position(z=STEP, relative=True, speed=SPEED, mvacc=ACC, wait=True)
    print("Move up code:", code)

    input("Press Enter to move back DOWN 20 mm...")
    code = arm.set_position(z=-STEP, relative=True, speed=SPEED, mvacc=ACC, wait=True)
    print("Move down code:", code)
except KeyboardInterrupt:
    print("Stopped by user")
    arm.set_state(4)
finally:
    print("Final position:", arm.get_position())
    arm.disconnect()
