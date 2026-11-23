from xarm.wrapper import XArmAPI

arm = XArmAPI('192.168.0.150')
print("Connected:", arm.connected)
print("Firmware:", arm.version)
print("State:", arm.state)
print("Error code:", arm.error_code, "| Warning code:", arm.warn_code)
print("Current position:", arm.get_position())
arm.disconnect()

