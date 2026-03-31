import time
from robot_interface import dobot


def hardware_test():
    print("=== HARDWARE-TEST ===")
    print("Verbindungsversuch...")
    if dobot.auto_connect("COM3"):
        print(" Verbunden!")
    else:
        print(" Verbindung fehlgeschlagen.")
        return False


    print("\nStarte Testbewegungen...")


    print(" Bewege zu X=210, Z=80")
    success = dobot.move(210, 0, 80, 0)
    print(f"   move() returned: {success}")
    time.sleep(2)


    print(" Bewege zurück zu X=200, Z=80")
    success = dobot.move(200, 0, 80, 0)
    print(f"   move() returned: {success}")
    time.sleep(2)


    print("\nGreifer-Test:")
    print(" Schließe Greifer")
    dobot.set_gripper(True)
    time.sleep(1)
    print(" Öffne Greifer")
    dobot.set_gripper(False)

    print("\nHardware-Test abgeschlossen.")
    return True


if __name__ == "__main__":
    hardware_test()