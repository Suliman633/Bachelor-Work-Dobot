import cv2
import numpy as np
from robot_interface import dobot
import signal
import sys
import time
from collections import deque
import math

try:
    from mediapipe import solutions as mp_solutions
    mp_hands    = mp_solutions.hands
    mp_drawing  = mp_solutions.drawing_utils
    print("MediaPipe erfolgreich geladen")
except ImportError:
    print("FEHLER: MediaPipe nicht installiert. pip install mediapipe")
    sys.exit(1)




def signal_handler(sig, frame):
    print("\nProgramm wird beendet...")
    try:
        if hasattr(dobot, "safe_disconnect"):
            dobot.safe_disconnect()
    except Exception as e:
        print(f"Disconnect-Fehler: {e}")
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def dobot_is_real_connected():
    try:
        return bool(dobot.is_connected()) if hasattr(dobot, "is_connected") else False
    except Exception:
        return False




class GestureClassifier:


    CONFIRM_DEFAULT = 4
    CONFIRM_STRICT  = 6

    def __init__(self):
        self._candidate: str | None = None
        self._count: int            = 0
        self._last_fired: dict      = {}

    @staticmethod
    def _up(lm, tip, pip) -> bool:
        return lm[tip].y < lm[pip].y - 0.015

    @staticmethod
    def _down(lm, tip, pip) -> bool:
        return lm[tip].y > lm[pip].y + 0.015

    @staticmethod
    def _dist2d(a, b) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def _others_folded(self, lm) -> bool:
        return (self._down(lm, 8,  6) and
                self._down(lm, 12, 10) and
                self._down(lm, 16, 14) and
                self._down(lm, 20, 18))

    def _raw(self, lm) -> str | None:
        ix = self._up(lm, 8,  6)
        mi = self._up(lm, 12, 10)
        ri = self._up(lm, 16, 14)
        pi = self._up(lm, 20, 18)


        if ix and pi and not mi and not ri:
            return "heavy_metal"


        if ix and mi and not ri and not pi:
            return "victory"


        if (lm[4].y < lm[3].y < lm[2].y and
                lm[4].y < lm[0].y - 0.05 and
                self._others_folded(lm)):
            return "thumbs_up"


        if (lm[4].y > lm[3].y > lm[2].y and
                lm[4].y > lm[0].y + 0.05 and
                self._others_folded(lm)):
            return "thumbs_down"


        if self._dist2d(lm[4], lm[8]) < 0.07 and sum(
                1 for f in [ix, mi, ri, pi] if not f) >= 4:
            return "fist"


        thumb = (abs(lm[4].x - lm[3].x) > 0.04 or lm[4].y < lm[3].y - 0.03)
        if sum([thumb, ix, mi, ri, pi]) >= 4:
            return "open_hand"

        return None

    def classify_discrete(self, lm, cooldown: float = 0.7) -> str | None:

        raw = self._raw(lm)

        if raw != self._candidate:
            self._candidate = raw
            self._count     = 1
        else:
            self._count += 1

        if raw is None:
            return None

        threshold = (self.CONFIRM_STRICT
                     if raw in ("fist", "open_hand")
                     else self.CONFIRM_DEFAULT)

        if self._count < threshold:
            return None

        now = time.time()
        if now - self._last_fired.get(raw, 0.0) < cooldown:
            return None

        self._last_fired[raw] = now
        return raw

    def current_raw(self, lm) -> str | None:

        return self._raw(lm)


class DynamicHandController:
    def __init__(self):
        self.current_x = 200.0
        self.current_y = 0.0
        self.current_z = 80.0

        self.x_limits = [140.0, 300.0]
        self.y_limits = [-135.0, 135.0]
        self.z_limits = [20.0, 200.0]

        self.initialized      = False
        self.skip_first_frame = False

        self.last_hand_x = 0.5
        self.last_hand_y = 0.5

        self.dy_buf = deque(maxlen=8)
        self.dz_buf = deque(maxlen=8)

        self.dead_y       = 0.004
        self.dead_z       = 0.004
        self.scale_y      = 360.0
        self.scale_z      = 360.0
        self.max_step     = 14.0
        self.smoothing    = 0.86
        self.min_move_dist = 0.35
        self.move_interval = 0.012

        self.last_move_time = 0.0
        self.gripper_closed = False
        self.near_boundary  = False

        self.last_left_action_time  = 0.0
        self.left_action_interval   = 0.08
        self.left_reference_x       = None
        self.left_initialized       = False

        self.side_step      = 4.0
        self.z_gesture_step = 4.0

        self.last_debug_time = 0.0

        print("DynamicHandController initialisiert")
        print(f"Start: X={self.current_x:.0f}, Y={self.current_y:.0f}, Z={self.current_z:.0f}")

    def initialize_hand_position(self, hand_landmarks):
        wrist = hand_landmarks.landmark[0]
        self.last_hand_x = wrist.x
        self.last_hand_y = wrist.y
        self.dy_buf.clear()
        self.dz_buf.clear()
        self.skip_first_frame = True
        self.initialized = True
        print(f"Bewegungs-Hand initialisiert: "
              f"X={self.current_x:.0f} Y={self.current_y:.0f} Z={self.current_z:.0f}")

    def should_move(self):
        return (time.time() - self.last_move_time) >= self.move_interval

    def update_move_time(self):
        self.last_move_time = time.time()

    @staticmethod
    def apply_deadzone(v, t):
        return v if abs(v) >= t else 0.0

    @staticmethod
    def clamp_step(v, m):
        return float(np.clip(v, -m, m))

    def map_movement_hand_to_robot_deltas(self, hand_landmarks):

        wrist = hand_landmarks.landmark[0]
        cx, cy = wrist.x, wrist.y

        if self.skip_first_frame:
            self.last_hand_x, self.last_hand_y = cx, cy
            self.skip_first_frame = False
            return 0.0, 0.0, 0.0

        raw_y = cx - self.last_hand_x
        raw_z = self.last_hand_y - cy

        if abs(raw_y) > 0.14 or abs(raw_z) > 0.14:
            self.last_hand_x, self.last_hand_y = cx, cy
            return 0.0, 0.0, 0.0

        self.dy_buf.append(raw_y)
        self.dz_buf.append(raw_z)

        ay = self.apply_deadzone(sum(self.dy_buf) / len(self.dy_buf), self.dead_y)
        az = self.apply_deadzone(sum(self.dz_buf) / len(self.dz_buf), self.dead_z)

        dy = self.clamp_step(-ay * self.scale_y, self.max_step)
        dz = self.clamp_step(az  * self.scale_z, self.max_step)

        self.last_hand_x, self.last_hand_y = cx, cy
        return 0.0, dy, dz

    def calculate_new_position(self, dx, dy, dz):
        sm = self.smoothing
        nx = float(np.clip(self.current_x + dx * sm, *self.x_limits))
        ny = float(np.clip(self.current_y + dy * sm, *self.y_limits))
        nz = float(np.clip(self.current_z + dz * sm, *self.z_limits))

        xm = min(abs(nx - self.x_limits[0]), abs(nx - self.x_limits[1]))
        ym = min(abs(ny - self.y_limits[0]), abs(ny - self.y_limits[1]))
        zm = min(abs(nz - self.z_limits[0]), abs(nz - self.z_limits[1]))
        self.near_boundary = (xm < 10) or (ym < 10) or (zm < 10)

        return nx, ny, nz



    def execute_discrete(self, gesture: str):
        if gesture == "gripper_close":
            if not self.gripper_closed:
                print("Greifer schließen")
                try:
                    if dobot.set_gripper(True):
                        self.gripper_closed = True
                except Exception as e:
                    print(f"Greifer-Fehler: {e}")

        elif gesture == "gripper_open":
            if self.gripper_closed:
                print("Greifer öffnen")
                try:
                    if dobot.set_gripper(False):
                        self.gripper_closed = False
                except Exception as e:
                    print(f"Greifer-Fehler: {e}")

        elif gesture == "home":
            print("Home-Position")
            self.current_x, self.current_y, self.current_z = 200.0, 0.0, 80.0
            if dobot_is_real_connected():
                try:
                    dobot.move(200.0, 0.0, 80.0, 0.0)
                    print("Home angefahren")
                except Exception as e:
                    print(f"Home-Fehler: {e}")
            self.initialized = False



    def process_left_continuous(self, hand_landmarks):
        now = time.time()
        if now - self.last_left_action_time < self.left_action_interval:
            return

        lm    = hand_landmarks.landmark
        wrist = lm[0]

        if not self.left_initialized:
            self.left_reference_x = wrist.x
            self.left_initialized = True


        if (lm[4].y < lm[3].y < lm[2].y and lm[4].y < lm[0].y - 0.05 and
                lm[8].y > lm[6].y and lm[12].y > lm[10].y and
                lm[16].y > lm[14].y and lm[20].y > lm[18].y):
            self._apply_continuous("z", +self.z_gesture_step)
            self.last_left_action_time = now
            return


        if (lm[4].y > lm[3].y > lm[2].y and lm[4].y > lm[0].y + 0.05 and
                lm[8].y > lm[6].y and lm[12].y > lm[10].y and
                lm[16].y > lm[14].y and lm[20].y > lm[18].y):
            self._apply_continuous("z", -self.z_gesture_step)
            self.last_left_action_time = now
            return


        ix = lm[8].y < lm[6].y - 0.015
        mi = lm[12].y > lm[10].y + 0.015
        ri = lm[16].y > lm[14].y + 0.015
        pi = lm[20].y < lm[18].y - 0.015

        if ix and pi and mi and ri:
            delta = wrist.x - self.left_reference_x
            if abs(delta) > 0.04:
                self._apply_continuous("y", -self.side_step if delta > 0 else +self.side_step)
                self.last_left_action_time = now

    def _apply_continuous(self, axis: str, delta: float):
        nx, ny, nz = self.current_x, self.current_y, self.current_z
        if axis == "z":
            nz = float(np.clip(nz + delta, *self.z_limits))
        elif axis == "y":
            ny = float(np.clip(ny + delta, *self.y_limits))

        if dobot_is_real_connected():
            try:
                if dobot.move(nx, ny, nz, 0.0):
                    self.current_x, self.current_y, self.current_z = nx, ny, nz
                    print(f"[{axis.upper()}] X={nx:.1f} Y={ny:.1f} Z={nz:.1f}")
            except Exception as e:
                print(f"Kontinuierlicher Move-Fehler: {e}")



FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTB = cv2.FONT_HERSHEY_DUPLEX


GESTURE_DISPLAY = {
    "fist":        ("Greifer ZU",         (60,  60,  220)),
    "open_hand":   ("Greifer AUF",        (60,  200, 60)),
    "victory":     ("HOME",               (40,  200, 220)),
    "thumbs_up":   ("Z  hoch  +4 mm",     (80,  220, 80)),
    "thumbs_down": ("Z  runter  -4 mm",   (80,  80,  220)),
    "heavy_metal": ("Y  links / rechts",  (220, 140, 40)),
}


def _alpha_rect(img, x1, y1, x2, y2, bgr, alpha=0.55):
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi  = img[y1:y2, x1:x2]
    fill = np.full(roi.shape, bgr, dtype=np.uint8)
    cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def draw_hud(frame, ctrl: DynamicHandController,
             active_gesture: str | None,
             move_active: bool, gesture_active: bool,
             debug: bool = False):

    h, w = frame.shape[:2]


    _alpha_rect(frame, 0, 0, w, 118, (10, 10, 20))

    conn = dobot_is_real_connected()
    cv2.putText(frame,
                f"DOBOT  {'VERBUNDEN' if conn else 'DEMO-MODUS'}",
                (14, 28), FONTB, 0.65,
                (60, 220, 60) if conn else (60, 60, 220), 1)

    cv2.putText(frame,
                f"X {ctrl.current_x:6.1f}   Y {ctrl.current_y:6.1f}   Z {ctrl.current_z:6.1f} mm",
                (14, 56), FONT, 0.55, (210, 210, 210), 1)

    gc = (60, 60, 220) if ctrl.gripper_closed else (60, 220, 60)
    cv2.putText(frame,
                f"Greifer: {'ZU' if ctrl.gripper_closed else 'OFFEN'}",
                (14, 82), FONT, 0.52, gc, 1)

    if ctrl.near_boundary:
        cv2.putText(frame, "NAHE GRENZE",
                    (w - 210, 82), FONT, 0.52, (0, 140, 255), 2)


    _alpha_rect(frame, 10,  125, 230, 153,
                (180, 60, 60) if move_active else (45, 45, 45), 0.65)
    _alpha_rect(frame, 240, 125, 480, 153,
                (60, 60, 180) if gesture_active else (45, 45, 45), 0.65)
    cv2.putText(frame, "RECHTS: YZ-Bewegung",
                (16, 145), FONT, 0.44, (240, 240, 240), 1)
    cv2.putText(frame, "LINKS: Gesten",
                (246, 145), FONT, 0.44, (240, 240, 240), 1)


    lx    = w - 250
    row_h = 22
    panel_h = len(GESTURE_DISPLAY) * row_h + 28
    _alpha_rect(frame, lx - 8, 0, w, panel_h, (12, 12, 22), 0.68)
    cv2.putText(frame, "GESTEN", (lx, 18), FONT, 0.38, (150, 150, 150), 1)

    for i, (g, (label, col)) in enumerate(GESTURE_DISPLAY.items()):
        yy = 34 + i * row_h
        if active_gesture == g:
            _alpha_rect(frame, lx - 6, yy - 14, w, yy + 8, col, 0.28)
            cv2.putText(frame, f"> {label}", (lx, yy), FONT, 0.40, col, 2)
        else:
            cv2.putText(frame, f"  {label}", (lx, yy), FONT, 0.38, col, 1)


    if active_gesture and active_gesture in GESTURE_DISPLAY:
        label, col = GESTURE_DISPLAY[active_gesture]
        _alpha_rect(frame, 0, h - 60, w, h, (8, 8, 18), 0.78)
        cv2.putText(frame, label, (14, h - 14), FONTB, 1.05, col, 2)


    if debug:
        cv2.putText(frame,
                    f"dy_buf:{len(ctrl.dy_buf)}  dz_buf:{len(ctrl.dz_buf)}"
                    f"  near:{ctrl.near_boundary}",
                    (14, h - 68), FONT, 0.36, (120, 120, 120), 1)

    return frame


def main():
    print("=" * 56)
    print("  Dobot Zwei-Hand-Gesten-Steuerung  v4.0")
    print("  RECHTE HAND = YZ-Tracking")
    print("  LINKE HAND  = Gesten-Kommandos")
    print("=" * 56)

    if not dobot.auto_connect("COM3"):
        print("Dobot nicht verbunden")
        return

    controller  = DynamicHandController()
    classifier  = GestureClassifier()
    debug       = False
    last_gesture_display = None
    last_gesture_time    = 0.0

    if dobot_is_real_connected():
        try:
            dobot.move(controller.current_x, controller.current_y,
                       controller.current_z, 0.0)
        except Exception as e:
            print(f"Startposition: {e}")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1,
    ) as hands:

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Kamera nicht gefunden")
            return

        print("\nSteuerung aktiv — 'q'/ESC=Ende  'd'=Debug\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.flip(frame, 1)
            fh, fw    = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = hands.process(rgb_frame)

            move_active    = False
            gesture_active = False

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lm, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness):

                    label = handedness.classification[0].label

                    is_movement = (label == "Left")
                    is_gesture  = (label == "Right")

                    dot_col = (220, 80, 80) if is_movement else (80, 80, 220)
                    mp_drawing.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=dot_col, thickness=3, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                    )

                    wx = int(hand_lm.landmark[0].x * fw)
                    wy = int(hand_lm.landmark[0].y * fh) - 22
                    cv2.putText(frame,
                                "RECHTE HAND: BEWEGUNG Y/Z" if is_movement else "LINKE HAND: GESTEN",
                                (wx, wy), FONT, 0.55, dot_col, 2)

                    lm = hand_lm.landmark


                    if is_movement:
                        move_active = True
                        if not controller.initialized:
                            controller.initialize_hand_position(hand_lm)

                        if controller.initialized and controller.should_move():
                            dx, dy, dz = controller.map_movement_hand_to_robot_deltas(hand_lm)
                            dist = math.sqrt(dy ** 2 + dz ** 2)

                            now = time.time()
                            if debug and now - controller.last_debug_time > 0.12:
                                print(f"DEBUG dy={dy:.2f} dz={dz:.2f} dist={dist:.2f}")
                                controller.last_debug_time = now

                            if dist >= controller.min_move_dist and dobot_is_real_connected():
                                nx, ny, nz = controller.calculate_new_position(dx, dy, dz)
                                try:
                                    if dobot.move(nx, ny, nz, 0.0):
                                        controller.current_x = nx
                                        controller.current_y = ny
                                        controller.current_z = nz
                                        controller.update_move_time()
                                except Exception as e:
                                    if debug:
                                        print(f"Move-Fehler: {e}")


                    elif is_gesture:
                        gesture_active = True


                        gesture = classifier.classify_discrete(lm, cooldown=0.7)
                        if gesture:
                            last_gesture_display = gesture
                            last_gesture_time    = time.time()

                            action_map = {
                                "fist":      "gripper_close",
                                "open_hand": "gripper_open",
                                "victory":   "home",
                            }
                            if gesture in action_map:
                                controller.execute_discrete(action_map[gesture])


                        controller.process_left_continuous(hand_lm)


                        raw = classifier.current_raw(lm)
                        if raw in ("thumbs_up", "thumbs_down", "heavy_metal"):
                            last_gesture_display = raw
                            last_gesture_time    = time.time()


            if not move_active:
                controller.initialized    = False
            if not gesture_active:
                controller.left_initialized = False


            if last_gesture_display and time.time() - last_gesture_time > 1.5:
                last_gesture_display = None

            frame = draw_hud(frame, controller,
                             last_gesture_display,
                             move_active, gesture_active,
                             debug)

            cv2.imshow("Dobot Gesten-Steuerung v4", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("d"):
                debug = not debug
                print(f"[Debug] {'AN' if debug else 'AUS'}")

    print("Beende...")
    cap.release()
    cv2.destroyAllWindows()
    try:
        if hasattr(dobot, "safe_disconnect"):
            dobot.safe_disconnect()
    except Exception:
        pass
    print("Fertig.")


if __name__ == "__main__":
    main()
