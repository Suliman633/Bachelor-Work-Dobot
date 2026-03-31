import os
import sys
import time
import math
import atexit
import threading

CURRENT_DIR      = os.path.dirname(os.path.abspath(__file__))
DOBOT_WRAPPER_DIR = os.path.join(CURRENT_DIR, "DobotDllType")
if DOBOT_WRAPPER_DIR not in sys.path:
    sys.path.append(DOBOT_WRAPPER_DIR)

import DobotDllType as dType




class DobotKinematics:
    def __init__(self):
        self.d1 = 138.0
        self.a2 = 135.0
        self.a3 = 147.0
        self.d4 = 77.0
        self.joint_limits = {
            "J1": [-135.0, 135.0],
            "J2": [0.0,    85.0],
            "J3": [-135.0, 10.0],
            "J4": [-90.0,  90.0],
        }
        print(f"Dobot Kinematik: d1={self.d1} a2={self.a2} a3={self.a3} d4={self.d4}")

    def inverse_kinematics(self, x, y, z, r=0.0):
        try:
            J1     = math.atan2(y, x)
            r_proj = math.sqrt(x * x + y * y)
            z_adj  = z - self.d1
            L      = math.sqrt(r_proj ** 2 + z_adj ** 2)
            if L > (self.a2 + self.a3) or L < abs(self.a2 - self.a3):
                return None
            cos_J3 = (L**2 - self.a2**2 - self.a3**2) / (2 * self.a2 * self.a3)
            cos_J3 = max(-1.0, min(1.0, cos_J3))
            J3     = -math.acos(cos_J3)
            k1     = self.a2 + self.a3 * math.cos(J3)
            k2     = self.a3 * math.sin(J3)
            J2     = math.atan2(z_adj, r_proj) - math.atan2(k2, k1)
            joints = {"J1": math.degrees(J1), "J2": math.degrees(J2),
                      "J3": math.degrees(J3), "J4": r}
            for joint, value in joints.items():
                lo, hi = self.joint_limits[joint]
                if not (lo <= value <= hi):
                    return None
            return joints
        except Exception:
            return None

    def is_reachable(self, x, y, z):
        return self.inverse_kinematics(x, y, z) is not None




class DobotMagician:
    def __init__(self):
        self.api       = None
        self.connected = False
        self.use_own_ik = False

        self._move_lock          = threading.Lock()
        self.last_move_time      = 0.0
        self.move_cooldown       = 0.015

        self.current_velocity    = 90
        self.current_acceleration = 90

        self.last_queued_index   = 0
        self.max_queue_ahead     = 3

        self.kinematics = DobotKinematics()

        self.safety_limits = {
            "x": [140, 300],
            "y": [-135, 135],
            "z": [20, 200],
            "r": [-80, 80],
        }

        self.last_target = None
        atexit.register(self.cleanup)
        print("DobotMagician Interface initialisiert")



    def cleanup(self):
        if self.connected:
            try:
                self.disable_all_end_effectors()
            except Exception:
                pass

    def _wait_for_cmd(self, cmd_index, timeout=10):
        start = time.time()
        while time.time() - start < timeout:
            try:
                if dType.GetQueuedCmdCurrentIndex(self.api)[0] >= cmd_index:
                    return True
            except Exception:
                pass
            time.sleep(0.03)
        return False

    def _queue_is_too_full(self):
        try:
            current = dType.GetQueuedCmdCurrentIndex(self.api)[0]
            return (self.last_queued_index - current) >= self.max_queue_ahead
        except Exception:
            return False

    def _print_alarms(self):
        try:
            raw, length = dType.GetAlarmsState(self.api)
            print(f"Alarme: {list(raw[:length])}")
        except Exception as e:
            print(f"Alarm-Lesefehler: {e}")

    def _print_pose(self, title="POSE"):
        try:
            p = dType.GetPose(self.api)
            print(f"{title}: X={p[0]:.1f} Y={p[1]:.1f} Z={p[2]:.1f} R={p[3]:.1f} "
                  f"| J1={p[4]:.1f} J2={p[5]:.1f} J3={p[6]:.1f} J4={p[7]:.1f}")
        except Exception as e:
            print(f"{title}: {e}")



    def auto_connect(self, preferred_port="COM3"):
        print("Starte Dobot-Verbindung...")
        old_cwd = os.getcwd()
        try:
            os.chdir(DOBOT_WRAPPER_DIR)
            self.api = dType.load()
        finally:
            os.chdir(old_cwd)

        try:
            devices = dType.SearchDobot(self.api)
            print("Verfügbare Geräte:", devices)
        except Exception:
            pass

        result = dType.ConnectDobot(self.api, preferred_port, 115200)
        print("ConnectDobot:", result)

        if result[0] == dType.DobotConnect.DobotConnect_NoError:
            self.connected = True
            print(f"Verbunden — MasterDevType={result[1]} SlaveDevType={result[2]}")
            print(f"Firmware: {result[3]} {result[4]}")
            self.initialize_robot()
            return True

        print("Verbindung fehlgeschlagen.")
        return False

    def initialize_robot(self):
        print("\n─── Initialisierung ───")

        dType.SetQueuedCmdClear(self.api)
        dType.SetQueuedCmdStartExec(self.api)


        print("Deaktiviere alle Endeffektoren...")
        self.disable_all_end_effectors()
        time.sleep(0.3)

        print("Alarme vor Clear:"); self._print_alarms()
        dType.ClearAllAlarmsState(self.api)
        time.sleep(0.3)
        print("Alarme nach Clear:"); self._print_alarms()

        self._print_pose("POSE VOR HOME")

        idx = dType.SetPTPCommonParams(
            self.api, self.current_velocity, self.current_acceleration, isQueued=1)[0]
        self._wait_for_cmd(idx, timeout=5)

        idx = dType.SetHOMEParams(self.api, 200, 0, 80, 0, isQueued=1)[0]
        self._wait_for_cmd(idx, timeout=5)


        home_result = dType.SetHOMECmd(self.api, 0, isQueued=1)
        self._wait_for_cmd(home_result[0], timeout=20)

        time.sleep(0.4)
        self._print_pose("POSE NACH HOME")


        self.disable_all_end_effectors()

        try:
            self.last_queued_index = dType.GetQueuedCmdCurrentIndex(self.api)[0]
        except Exception:
            self.last_queued_index = 0

        print("─── Initialisierung abgeschlossen ───\n")



    def disable_all_end_effectors(self):

        if not self.connected:
            return False

        ok = True


        for attempt in range(3):
            try:
                dType.SetEndEffectorSuctionCup(self.api, True, False, isQueued=0)
                time.sleep(0.12)

                state = dType.GetEndEffectorSuctionCup(self.api)
                print(f"Suction OFF (Versuch {attempt+1}): isOn={state}")
                break
            except Exception as e:
                ok = False
                print(f"Suction-OFF Fehler (Versuch {attempt+1}): {e}")
                time.sleep(0.1)


        try:
            dType.SetEndEffectorGripper(self.api, True, False, isQueued=0)
            time.sleep(0.12)
            state = dType.GetEndEffectorGripper(self.api)
            print(f"Gripper OFF: isOn={state}")
        except Exception as e:
            ok = False
            print(f"Gripper-OFF Fehler: {e}")


        try:
            dType.SetEndEffectorLaser(self.api, True, False, isQueued=0)
            time.sleep(0.05)
        except Exception:
            pass

        return ok



    def move_xyz(self, x, y, z, r=0):
        with self._move_lock:
            now = time.time()
            if now - self.last_move_time < self.move_cooldown:
                return False
            if self._queue_is_too_full():
                return False

            x = float(max(self.safety_limits["x"][0], min(self.safety_limits["x"][1], x)))
            y = float(max(self.safety_limits["y"][0], min(self.safety_limits["y"][1], y)))
            z = float(max(self.safety_limits["z"][0], min(self.safety_limits["z"][1], z)))
            r = float(max(self.safety_limits["r"][0], min(self.safety_limits["r"][1], r)))

            if self.use_own_ik and not self.kinematics.is_reachable(x, y, z):
                print(f"IK: Ziel nicht erreichbar X={x:.1f} Y={y:.1f} Z={z:.1f}")
                return False

            new_target = (round(x, 1), round(y, 1), round(z, 1), round(r, 1))
            if self.last_target == new_target:
                return False
            self.last_target = new_target

            try:
                cmd_idx = dType.SetPTPCmd(
                    self.api,
                    dType.PTPMode.PTPMOVJXYZMode,
                    x, y, z, r,
                    isQueued=1,
                )[0]
                self.last_queued_index = cmd_idx
                self.last_move_time    = now
                return True
            except Exception as e:
                print(f"move_xyz Fehler: {e}")
                return False

    def move(self, x, y, z, r=0):
        return self.move_xyz(x, y, z, r)



    def set_suction(self, on: bool):
        if not self.connected:
            return False
        try:
            dType.SetEndEffectorSuctionCup(self.api, True, on, isQueued=0)
            time.sleep(0.15)
            state = dType.GetEndEffectorSuctionCup(self.api)
            print(f"Suction {'AN' if on else 'AUS'}: {state}")
            return True
        except Exception as e:
            print(f"Suction-Fehler: {e}")
            return False



    def set_gripper(self, close: bool):

        if not self.connected:
            return False
        try:

            dType.SetEndEffectorSuctionCup(self.api, True, False, isQueued=0)
            time.sleep(0.15)

            dType.SetEndEffectorGripper(self.api, True, close, isQueued=0)
            time.sleep(0.15)

            s = dType.GetEndEffectorSuctionCup(self.api)
            g = dType.GetEndEffectorGripper(self.api)
            print(f"Gripper {'ZU' if close else 'AUF'} — Suction:{s} Gripper:{g}")
            return True
        except Exception as e:
            print(f"Greifer-Fehler: {e}")
            return False



    def is_connected(self):
        if not self.connected or self.api is None:
            return False
        try:
            dType.GetPose(self.api)
            return True
        except Exception:
            return False

    def set_movement_speed(self, velocity_percent, acceleration_percent):
        self.current_velocity     = max(10, min(100, velocity_percent))
        self.current_acceleration = max(10, min(100, acceleration_percent))
        if not self.connected:
            return False
        idx = dType.SetPTPCommonParams(
            self.api, self.current_velocity, self.current_acceleration, isQueued=1)[0]
        return self._wait_for_cmd(idx, timeout=5)



    def safe_disconnect(self):
        if self.connected:
            print("Trenne Dobot...")


            for attempt in range(3):
                try:
                    self.disable_all_end_effectors()
                    break
                except Exception as e:
                    print(f"disable Versuch {attempt+1}: {e}")
                    time.sleep(0.1)

            time.sleep(0.3)
            try:
                dType.DisconnectDobot(self.api)
            except Exception:
                pass
            self.connected = False
            print("Getrennt.")




dobot = DobotMagician()
__all__ = ["dobot", "DobotMagician"]
