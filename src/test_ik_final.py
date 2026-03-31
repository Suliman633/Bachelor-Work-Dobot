
import math
from robot_interface import dobot

print("\n" + "=" * 60)
print(" TESTE KORRIGIERTE KINEMATIK")
print("=" * 60)


test_points = [
    (200, 0, 80, "Home"),
    (180, 30, 70, "Rechts"),
    (150, 0, 50, "Nah"),
]

for x, y, z, name in test_points:
    print(f"\n {name}: X={x}, Y={y}, Z={z}")

    # Berechne IK
    joints = dobot.kinematics.inverse_kinematics(x, y, z, 0)

    if joints:
        print(f"   J1={joints['J1']:6.1f}°")
        print(f"   J2={joints['J2']:6.1f}°")
        print(f"   J3={joints['J3']:6.1f}°")
        print(f"   J4={joints['J4']:6.1f}°")


        x_fk, y_fk, z_fk = dobot.kinematics.forward_kinematics(
            joints['J1'], joints['J2'], joints['J3'], joints['J4']
        )

        error = math.sqrt((x - x_fk) ** 2 + (y - y_fk) ** 2 + (z - z_fk) ** 2)
        print(f"   Rückrechnung: X={x_fk:.1f}, Y={y_fk:.1f}, Z={z_fk:.1f}")
        print(f"   Fehler: {error:.3f} mm")

        if error < 5.0:
            print(f"    IK funktioniert!")
        else:
            print(f"    Fehler zu groß!")
    else:
        print(f"    Keine Lösung")