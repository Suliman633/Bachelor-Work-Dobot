
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_interface import dobot

print("="*50)
print("  DIREKTER HARDWARE-TEST v2")
print("="*50)

dobot.auto_connect('COM3')
time.sleep(1)

if not dobot.is_connected():
    print("FEHLER: Nicht verbunden!")
    sys.exit(1)

print("Verbunden!\n")
input("ENTER zum Starten (Roboter bewegt sich!)...")

tests = [
    ("Home",        200,   0,  80),
    ("Hoch Z=120",  200,   0, 120),
    ("Home",        200,   0,  80),
    ("Links Y=+30", 200,  30,  80),
    ("Home",        200,   0,  80),
    ("Rechts Y=-30",200, -30,  80),
    ("Home",        200,   0,  80),
]

for name, x, y, z in tests:
    print(f"\n-> {name} (X={x}, Y={y}, Z={z})")
    ok = dobot.move(x, y, z)
    print(f"   Ergebnis: {'OK' if ok else 'FEHLER'}")
    time.sleep(3)

print("\n\nHat sich der Roboter bewegt? (ja/nein)")
antwort = input("> ").strip().lower()

if antwort != 'ja':
    print("\nVersuche DLL-interne IK (Modus PTPMOVJXYZMode)...")
    input("ENTER...")
    for name, x, y, z in tests:
        print(f"\n-> {name} XYZ-DLL")
        dobot.move_xyz_dll(x, y, z)
        time.sleep(3)

print("\nTest beendet!")
dobot.safe_disconnect()
