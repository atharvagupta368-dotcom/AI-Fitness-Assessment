
import cv2
import mediapipe as mp
import math

VIDEO_PATH = "squat.mp4"
MODEL_PATH = "pose_landmarker_lite.task"

DOWN_ANGLE = 115
UP_ANGLE = 155
MIN_VISIBILITY = 0.5


def calculate_angle(a, b, c):
    """Calculate angle ABC in degrees; B is the joint."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba == 0 or mag_bc == 0:
        return None

    cosine = dot / (mag_ba * mag_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def torso_lean_from_vertical(shoulder, hip):
    """
    Estimate torso angle from vertical in image coordinates.
    This is a rough 2D estimate, not a medical posture metric.
    """
    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]

    if dy == 0:
        return 90.0

    return abs(math.degrees(math.atan2(dx, dy)))


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32)
]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise SystemExit("Cannot open squat.mp4. Check the filename and location.")

fps = video.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

reps = 0
is_down = False
frame_number = 0

try:
    with PoseLandmarker.create_from_options(options) as landmarker:
        cv2.namedWindow("AI Fitness Assessment", cv2.WINDOW_NORMAL)

        while True:
            success, frame = video.read()
            if not success:
                break

            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(frame_number * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            angle = None
            torso_lean = None
            feedback = "Move into view"
            movement = "NOT DETECTED"
            score = None

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # Draw body connections
                for start, end in CONNECTIONS:
                    p1, p2 = landmarks[start], landmarks[end]

                    if (
                        p1.visibility > MIN_VISIBILITY
                        and p2.visibility > MIN_VISIBILITY
                    ):
                        cv2.line(
                            frame,
                            (int(p1.x * width), int(p1.y * height)),
                            (int(p2.x * width), int(p2.y * height)),
                            (0, 255, 0), 3
                        )

                # Draw visible landmarks
                for point in landmarks:
                    if point.visibility > MIN_VISIBILITY:
                        cv2.circle(
                            frame,
                            (int(point.x * width), int(point.y * height)),
                            4, (0, 0, 255), -1
                        )

                # Right-side hip, knee, ankle and shoulder
                shoulder = landmarks[12]
                hip = landmarks[24]
                knee = landmarks[26]
                ankle = landmarks[28]

                visible = min(
                    shoulder.visibility,
                    hip.visibility,
                    knee.visibility,
                    ankle.visibility
                ) > MIN_VISIBILITY

                if visible:
                    angle = calculate_angle(
                        (hip.x, hip.y),
                        (knee.x, knee.y),
                        (ankle.x, ankle.y)
                    )

                    torso_lean = torso_lean_from_vertical(
                        (shoulder.x, shoulder.y),
                        (hip.x, hip.y)
                    )

                    if angle is not None:
                        if angle < DOWN_ANGLE:
                            movement = "SQUATTING"
                            is_down = True
                        elif angle > UP_ANGLE:
                            movement = "RISING" if is_down else "STANDING"

                            if is_down:
                                reps += 1
                                is_down = False
                        else:
                            movement = "RISING" if is_down else "MOVING"

                        # Basic heuristic feedback
                        feedback_items = []
                        score = 100

                        if angle < 55:
                            feedback_items.append("Very deep knee bend")
                            score -= 20

                        if torso_lean > 45:
                            feedback_items.append("Check torso position")
                            score -= 20

                        if angle < DOWN_ANGLE and torso_lean > 45:
                            feedback_items.append("Control your movement")
                            score -= 10

                        if not feedback_items:
                            feedback = "Movement looks consistent"
                        else:
                            feedback = feedback_items[0]

                        score = max(0, score)

            # Dashboard overlays
            cv2.putText(
                frame, f"Squat Reps: {reps}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2
            )

            cv2.putText(
                frame, f"Movement: {movement}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2
            )

            if angle is not None:
                cv2.putText(
                    frame, f"Knee Angle: {angle:.1f} deg",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2
                )

            if torso_lean is not None:
                cv2.putText(
                    frame, f"Torso Angle: {torso_lean:.1f} deg",
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 200, 0), 2
                )

            if score is not None:
                cv2.putText(
                    frame, f"Rule Score: {score}/100",
                    (20, 175), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2
                )

            cv2.putText(
                frame, f"Feedback: {feedback}",
                (20, 210), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2
            )

            # Fit full frame inside display
            scale = min(900 / width, 600 / height)
            display_frame = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale)))
            )

            cv2.imshow("AI Fitness Assessment", display_frame)

            frame_number += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    video.release()
    cv2.destroyAllWindows()
