
import math
import os
import time

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


MODEL_PATH = "pose_landmarker_lite.task"
DOWN_ANGLE = 115
UP_ANGLE = 155
VISIBILITY_THRESHOLD = 0.5

st.set_page_config(
    page_title="AI Fitness Assessment",
    page_icon="🏋️",
    layout="wide",
)


def calculate_angle(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return None

    cosine = np.dot(ba, bc) / denominator
    return math.degrees(
        math.acos(np.clip(cosine, -1.0, 1.0))
    )


def torso_lean_from_vertical(shoulder, hip):
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]

    if dx == 0 and dy == 0:
        return None

    return abs(math.degrees(math.atan2(dx, -dy)))


def is_visible(landmark):
    return landmark.visibility >= VISIBILITY_THRESHOLD


def draw_skeleton(frame, landmarks):
    height, width = frame.shape[:2]

    connections = [
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 23), (12, 24),
        (23, 24),
        (23, 25), (25, 27),
        (24, 26), (26, 28),
        (27, 29), (29, 31),
        (28, 30), (30, 32),
    ]

    for start, end in connections:
        if is_visible(landmarks[start]) and is_visible(landmarks[end]):
            x1 = int(landmarks[start].x * width)
            y1 = int(landmarks[start].y * height)
            x2 = int(landmarks[end].x * width)
            y2 = int(landmarks[end].y * height)

            cv2.line(
                frame, (x1, y1), (x2, y2),
                (60, 220, 100), 3, cv2.LINE_AA
            )

    for point in landmarks:
        if is_visible(point):
            x = int(point.x * width)
            y = int(point.y * height)
            cv2.circle(
                frame, (x, y), 4,
                (0, 220, 255), -1, cv2.LINE_AA
            )


def draw_label(frame, text, y, color=(255, 255, 255)):
    cv2.putText(
        frame, text, (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, color, 2, cv2.LINE_AA
    )


class FitnessProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = __import__("threading").Lock()
        self.landmarker = None
        self.timestamp_ms = 0
        self.reps = 0
        self.is_down = False
        self.previous_angle = None
        self.movement = "WAITING"
        self.knee_angle = None
        self.torso_angle = None
        self.status = "POSITION YOUR BODY IN VIEW"
        self.feedback = "Keep your full body visible."
        self.score = 100
        self.pose_detected = False

    def ensure_landmarker(self):
        if self.landmarker is None:
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=MODEL_PATH
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
            )
            self.landmarker = (
                mp.tasks.vision.PoseLandmarker
                .create_from_options(options)
            )

    def recv(self, video_frame):
        frame = video_frame.to_ndarray(format="bgr24")
        output = frame.copy()

        if not os.path.exists(MODEL_PATH):
            draw_label(output, "POSE MODEL FILE NOT FOUND", 40, (0, 0, 255))
            return av.VideoFrame.from_ndarray(output, format="bgr24")

        try:
            self.ensure_landmarker()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            self.timestamp_ms += 33
            result = self.landmarker.detect_for_video(
                mp_image, self.timestamp_ms
            )

            self.knee_angle = None
            self.torso_angle = None
            self.pose_detected = False
            self.movement = "NOT DETECTED"
            self.status = "CORRECT YOUR POSTURE"
            self.feedback = "Step into view and show your full body."
            self.score = 0

            if result.pose_landmarks:
                points = result.pose_landmarks[0]
                draw_skeleton(output, points)

                required = [12, 24, 26, 28]

                if all(is_visible(points[i]) for i in required):
                    self.pose_detected = True

                    shoulder = (points[12].x, points[12].y)
                    hip = (points[24].x, points[24].y)
                    knee = (points[26].x, points[26].y)
                    ankle = (points[28].x, points[28].y)

                    self.knee_angle = calculate_angle(
                        hip, knee, ankle
                    )
                    self.torso_angle = torso_lean_from_vertical(
                        shoulder, hip
                    )

                    issues = []

                    if self.knee_angle is not None:
                        if self.knee_angle < DOWN_ANGLE:
                            self.is_down = True
                            self.movement = "SQUATTING"
                        elif self.knee_angle > UP_ANGLE:
                            if self.is_down:
                                self.reps += 1
                                self.is_down = False
                            self.movement = "STANDING"
                        elif (
                            self.previous_angle is not None
                            and self.knee_angle < self.previous_angle
                        ):
                            self.movement = "DESCENDING"
                        else:
                            self.movement = "RISING"

                    # Simple illustrative rules, not medical standards.
                    # Torso lean is most meaningful during squat movement.
                    if (
                        self.torso_angle is not None
                        and self.movement in (
                            "SQUATTING", "DESCENDING", "RISING"
                        )
                        and self.torso_angle > 40
                    ):
                        issues.append(
                            "Excessive torso lean detected."
                        )

                    if (
                        self.knee_angle is not None
                        and self.knee_angle < 40
                    ):
                        issues.append(
                            "Unusually small knee angle detected."
                        )

                    self.score = max(0, 100 - 20 * len(issues))

                    if issues:
                        self.status = "CORRECT YOUR POSTURE"
                        self.feedback = " ".join(issues)
                    else:
                        self.status = "POSTURE OK"
                        self.feedback = (
                            "No configured warning detected. "
                            "This does not guarantee correct form."
                        )

                    self.previous_angle = self.knee_angle

                else:
                    self.status = "CORRECT YOUR POSTURE"
                    self.feedback = (
                        "Body landmarks are unclear. "
                        "Move back and keep your hips, knees and ankles visible."
                    )

            draw_label(
                output,
                f"REPS: {self.reps}",
                35,
                (80, 255, 120),
            )
            draw_label(
                output,
                self.status,
                70,
                (0, 0, 255)
                if self.status == "CORRECT YOUR POSTURE"
                else (80, 255, 120),
            )

            if self.knee_angle is not None:
                draw_label(
                    output,
                    f"KNEE ANGLE: {self.knee_angle:.1f} deg",
                    105,
                )

            if self.torso_angle is not None:
                draw_label(
                    output,
                    f"TORSO LEAN: {self.torso_angle:.1f} deg",
                    140,
                )

        except Exception as error:
            self.status = "CORRECT YOUR POSTURE"
            self.feedback = f"Camera processing error: {error}"
            draw_label(output, "PROCESSING ERROR", 40, (0, 0, 255))

        return av.VideoFrame.from_ndarray(output, format="bgr24")

    def get_metrics(self):
        return {
            "reps": self.reps,
            "movement": self.movement,
            "knee_angle": self.knee_angle,
            "torso_angle": self.torso_angle,
            "status": self.status,
            "feedback": self.feedback,
            "score": self.score,
            "pose_detected": self.pose_detected,
        }

    def stop(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None


# ---------------- USER INTERFACE ----------------

st.title("🏋️ AI Fitness & Posture Assessment")
st.write(
    "Live camera-based squat tracking and rule-based posture feedback."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Missing {MODEL_PATH}. Put the model file in this project folder."
    )
    st.stop()

st.warning(
    "Camera access is required. Allow camera permission in your browser. "
    "Use good lighting and keep your full body visible."
)

with st.sidebar:
    st.header("Assessment Settings")
    st.write("Exercise: **Bodyweight squat**")
    st.write(f"Squat-down threshold: {DOWN_ANGLE}°")
    st.write(f"Standing threshold: {UP_ANGLE}°")
    st.divider()
    st.caption(
        "Educational prototype only. The rules are not clinically "
        "validated and cannot reliably detect every posture error."
    )

ctx = webrtc_streamer(
    key="live-fitness-camera",
    video_processor_factory=FitnessProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)

st.divider()
st.subheader("Live Assessment Results")

c1, c2, c3, c4 = st.columns(4)
rep_box = c1.empty()
movement_box = c2.empty()
knee_box = c3.empty()
score_box = c4.empty()

status_box = st.empty()
feedback_box = st.empty()
torso_box = st.empty()

if ctx.state.playing:
    @st.fragment(run_every="500ms")
    def update_dashboard():
        processor = ctx.video_processor

        if processor is None:
            st.info("Starting pose detection...")
            return

        data = processor.get_metrics()

        rep_box.metric("Repetitions", data["reps"])
        movement_box.metric("Movement", data["movement"])

        knee_box.metric(
            "Knee angle",
            f'{data["knee_angle"]:.1f}°'
            if data["knee_angle"] is not None else "N/A",
        )

        score_box.metric("Rule score", f'{data["score"]}/100')

        torso_box.metric(
            "Torso angle",
            f'{data["torso_angle"]:.1f}°'
            if data["torso_angle"] is not None else "N/A",
        )

        if data["status"] == "POSTURE OK":
            status_box.success("✅ POSTURE OK")
        else:
            status_box.error("⚠️ CORRECT YOUR POSTURE")

        feedback_box.write(data["feedback"])

    update_dashboard()
else:
    st.info("Click START in the camera panel above to begin.")
