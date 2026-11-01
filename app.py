
import cv2
import time
import tempfile
import numpy as np
import streamlit as st
import mediapipe as mp

from pathlib import Path
from collections import deque


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Fitness Assessment",
    page_icon="💪",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"

EXERCISES = [
    "Squats",
    "Push-ups",
    "Bicep Curls",
    "Lateral Raises"
]

VISIBILITY_THRESHOLD = 0.5

# Landmark indexes
LS, RS = 11, 12
LE, RE = 13, 14
LW, RW = 15, 16
LH, RH = 23, 24
LK, RK = 25, 26
LA, RA = 27, 28

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (11, 23),
    (12, 24), (23, 24), (23, 25),
    (25, 27), (24, 26), (26, 28),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22)
]

THRESHOLDS = {
    "Squats": {"down": 115, "up": 155},
    "Push-ups": {"down": 90, "up": 150},
    "Bicep Curls": {"up": 60, "down": 140},
    "Lateral Raises": {"up": 55, "down": 25}
}


# =========================================================
# ANGLE CALCULATIONS
# =========================================================

def calculate_angle(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-8:
        return None

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def get_point(landmark):
    return (landmark.x, landmark.y)


def visible(landmarks, indices):
    return all(
        landmarks[i].visibility >= VISIBILITY_THRESHOLD
        for i in indices
    )


def best_angle(landmarks, sides):
    candidates = []

    for ids in sides:
        if not visible(landmarks, ids):
            continue

        angle = calculate_angle(*[
            get_point(landmarks[i]) for i in ids
        ])

        if angle is not None:
            confidence = min(
                landmarks[i].visibility for i in ids
            )
            candidates.append((angle, confidence))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[1])[0]


# =========================================================
# EXERCISE MEASUREMENTS
# =========================================================

def get_measurement(landmarks, exercise):

    if exercise == "Squats":
        return best_angle(
            landmarks,
            [(LH, LK, LA), (RH, RK, RA)]
        )

    if exercise in ("Push-ups", "Bicep Curls"):
        return best_angle(
            landmarks,
            [(LS, LE, LW), (RS, RE, RW)]
        )

    return best_angle(
        landmarks,
        [(LH, LS, LE), (RH, RS, RE)]
    )


def get_torso_inclination(landmarks):
    """
    Estimate torso inclination relative to vertical.
    Best suited to a side-view squat video.
    """

    if not visible(landmarks, [LS, RS, LH, RH]):
        return None

    shoulder_x = (
        landmarks[LS].x + landmarks[RS].x
    ) / 2

    shoulder_y = (
        landmarks[LS].y + landmarks[RS].y
    ) / 2

    hip_x = (
        landmarks[LH].x + landmarks[RH].x
    ) / 2

    hip_y = (
        landmarks[LH].y + landmarks[RH].y
    ) / 2

    dx = shoulder_x - hip_x
    dy = shoulder_y - hip_y

    if abs(dy) < 1e-6:
        return 90.0

    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


# =========================================================
# MOVEMENT CLASSIFICATION
# =========================================================

def classify_movement(angle, exercise):

    if angle is None:
        return "NOT DETECTED"

    t = THRESHOLDS[exercise]

    if exercise in ("Squats", "Push-ups"):

        if angle <= t["down"]:
            return "DOWN"

        if angle >= t["up"]:
            return "UP"

    elif exercise == "Bicep Curls":

        if angle <= t["up"]:
            return "UP"

        if angle >= t["down"]:
            return "DOWN"

    elif exercise == "Lateral Raises":

        if angle >= t["up"]:
            return "UP"

        if angle <= t["down"]:
            return "DOWN"

    return "MOVING"


# =========================================================
# SQUAT POSTURE ASSESSMENT
# =========================================================

def assess_squat_posture(landmarks, knee_angle):

    if knee_angle is None:
        return False, "Body landmarks not detected"

    torso_angle = get_torso_inclination(landmarks)

    if torso_angle is None:
        return False, "Keep your upper body visible"

    # Excessive torso inclination warning.
    # Initial threshold: validate using your own video.
    if torso_angle > 55:
        return False, "Keep your torso more upright"

    # Squat depth is evaluated when the knees bend.
    if knee_angle < 115:
        return True, "Good squat depth"

    # Standing position is not considered a squat-depth error.
    if knee_angle >= 155:
        return True, "Good standing position"

    # During the movement, depth cannot be judged yet.
    return True, "Movement detected"


def assess_posture(landmarks, angle, exercise):

    if exercise == "Squats":
        return assess_squat_posture(landmarks, angle)

    # Other exercises currently have pose tracking,
    # but not validated red/green posture rules.
    return None, "Posture assessment not configured"


# =========================================================
# REPETITION COUNTER
# =========================================================

def new_counter():

    return {
        "phase": "WAITING",
        "reps": 0,
        "last_count_time": -1.0
    }


def update_counter(state, counter, timestamp):

    if state == "DOWN":
        counter["phase"] = "DOWN"

    elif state == "UP" and counter["phase"] == "DOWN":

        if timestamp - counter["last_count_time"] >= 0.35:
            counter["reps"] += 1
            counter["last_count_time"] = timestamp

        counter["phase"] = "UP"

    return counter


# =========================================================
# DRAW SKELETON
# =========================================================

def draw_pose(frame, landmarks, posture_good):

    height, width = frame.shape[:2]

    points = {}

    for i, lm in enumerate(landmarks):

        points[i] = (
            int(lm.x * width),
            int(lm.y * height)
        )

    if posture_good is True:
        color = (0, 255, 0)       # GREEN

    elif posture_good is False:
        color = (0, 0, 255)       # RED

    else:
        color = (0, 165, 255)     # ORANGE: not assessed

    for start, end in CONNECTIONS:

        if start >= len(landmarks) or end >= len(landmarks):
            continue

        if (
            landmarks[start].visibility >= VISIBILITY_THRESHOLD
            and landmarks[end].visibility >= VISIBILITY_THRESHOLD
        ):

            cv2.line(
                frame,
                points[start],
                points[end],
                color,
                3
            )

    for i, point in points.items():

        if landmarks[i].visibility >= VISIBILITY_THRESHOLD:

            cv2.circle(
                frame,
                point,
                5,
                color,
                -1
            )

    return frame


# =========================================================
# VIDEO OVERLAY
# =========================================================

def draw_overlay(
    frame,
    exercise,
    reps,
    state,
    posture_good,
    feedback
):

    cv2.rectangle(
        frame,
        (10, 10),
        (560, 180),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        frame,
        f"Exercise: {exercise}",
        (22, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Repetitions: {reps}",
        (22, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Movement: {state}",
        (22, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 220, 255),
        2
    )

    if posture_good is True:
        color = (0, 255, 0)
        label = "POSTURE: CORRECT"

    elif posture_good is False:
        color = (0, 0, 255)
        label = "POSTURE: INCORRECT"

    else:
        color = (0, 165, 255)
        label = "POSTURE: NOT ASSESSED"

    cv2.putText(
        frame,
        label,
        (22, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )

    cv2.putText(
        frame,
        feedback[:55],
        (22, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    return frame


# =========================================================
# VIDEO PROCESSING
# =========================================================

def process_video(input_path, output_path, exercise, progress):

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError("Cannot open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Unable to create output video.")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    counter = new_counter()

    processed_frames = 0
    visible_frames = 0

    correct_frames = 0
    incorrect_frames = 0
    unassessed_frames = 0

    last_state = "NOT DETECTED"
    last_feedback = "Waiting for pose"

    angle_history = deque(maxlen=5)

    start_time = time.perf_counter()

    try:

        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:

                ok, frame = cap.read()

                if not ok:
                    break

                processed_frames += 1

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb)
                )

                timestamp_ms = int(
                    (processed_frames - 1) * 1000 / fps
                )

                result = landmarker.detect_for_video(
                    image,
                    timestamp_ms
                )

                timestamp_seconds = (
                    processed_frames - 1
                ) / fps

                posture_good = None

                if result.pose_landmarks:

                    landmarks = result.pose_landmarks[0]
                    visible_frames += 1

                    angle = get_measurement(
                        landmarks,
                        exercise
                    )

                    if angle is not None:

                        angle_history.append(angle)

                        smoothed_angle = float(
                            np.median(angle_history)
                        )

                        last_state = classify_movement(
                            smoothed_angle,
                            exercise
                        )

                        counter = update_counter(
                            last_state,
                            counter,
                            timestamp_seconds
                        )

                        posture_good, last_feedback = assess_posture(
                            landmarks,
                            smoothed_angle,
                            exercise
                        )

                    else:
                        last_state = "NOT DETECTED"
                        last_feedback = "Relevant joints not visible"

                    if posture_good is True:
                        correct_frames += 1

                    elif posture_good is False:
                        incorrect_frames += 1

                    else:
                        unassessed_frames += 1

                    draw_pose(
                        frame,
                        landmarks,
                        posture_good
                    )

                else:

                    last_state = "NOT DETECTED"
                    last_feedback = "Person not detected"

                    angle_history.clear()
                    unassessed_frames += 1

                draw_overlay(
                    frame,
                    exercise,
                    counter["reps"],
                    last_state,
                    posture_good,
                    last_feedback
                )

                writer.write(frame)

                if total_frames > 0:

                    progress.progress(
                        min(
                            processed_frames / total_frames,
                            1.0
                        )
                    )

    finally:

        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start_time

    duration = processed_frames / fps

    detection_rate = (
        100 * visible_frames / processed_frames
        if processed_frames else 0.0
    )

    assessed_frames = correct_frames + incorrect_frames

    posture_score = (
        100 * correct_frames / assessed_frames
        if assessed_frames else 0.0
    )

    return {
        "exercise": exercise,
        "reps": counter["reps"],
        "frames": processed_frames,
        "visible": visible_frames,
        "detection_rate": detection_rate,
        "duration": duration,
        "processing_time": elapsed,
        "last_state": last_state,
        "correct_frames": correct_frames,
        "incorrect_frames": incorrect_frames,
        "unassessed_frames": unassessed_frames,
        "posture_score": posture_score,
        "assessed_frames": assessed_frames
    }


# =========================================================
# STREAMLIT DASHBOARD
# =========================================================

st.title("💪 AI Fitness Assessment")

st.write(
    "Upload a recorded exercise video to analyze movement, "
    "estimate repetitions and visualize body posture."
)

if not MODEL_PATH.exists():

    st.error(
        "Missing pose_landmarker_lite.task. "
        "Place the model file beside app.py."
    )

    st.stop()


with st.sidebar:

    st.header("Workout Settings")

    exercise = st.selectbox(
        "Choose exercise",
        EXERCISES
    )

    st.info(
        "Red/green posture assessment is currently "
        "configured for squats. Other exercises show "
        "pose tracking without a validated form score."
    )


uploaded = st.file_uploader(
    "Upload workout video",
    type=["mp4", "mov", "avi", "mkv"]
)


if uploaded:

    st.subheader("Original Video")
    st.video(uploaded)

    if st.button("Analyze Workout", type="primary"):

        progress = st.progress(0)
        status = st.empty()

        suffix = Path(uploaded.name).suffix or ".mp4"

        try:

            with tempfile.TemporaryDirectory() as temp:

                input_path = Path(temp) / f"input{suffix}"

                output_path = (
                    Path(temp) / "assessed_video.mp4"
                )

                input_path.write_bytes(
                    uploaded.getvalue()
                )

                status.info("Analyzing your exercise video...")

                report = process_video(
                    input_path,
                    output_path,
                    exercise,
                    progress
                )

                video_bytes = output_path.read_bytes()

                if not video_bytes:
                    raise RuntimeError(
                        "Processed video is empty."
                    )

                st.session_state["fitness_report"] = report
                st.session_state["fitness_video"] = video_bytes

            status.success("Analysis completed successfully!")

        except Exception as exc:

            status.error(
                f"Analysis failed: {exc}"
            )


# =========================================================
# ASSESSMENT REPORT
# =========================================================

if (
    "fitness_report" in st.session_state
    and "fitness_video" in st.session_state
):

    report = st.session_state["fitness_report"]

    video_bytes = st.session_state["fitness_video"]

    st.divider()

    st.header("📊 Assessment Report")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Estimated Repetitions",
        report["reps"]
    )

    c2.metric(
        "Pose Detection Rate",
        f'{report["detection_rate"]:.1f}%'
    )

    duration = int(report["duration"])

    c3.metric(
        "Video Duration",
        f"{duration // 60:02d}:{duration % 60:02d}"
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "Exercise",
        report["exercise"]
    )

    c5.metric(
        "Frames Processed",
        report["frames"]
    )

    c6.metric(
        "Processing Time",
        f'{report["processing_time"]:.1f} sec'
    )

    st.subheader("Posture Assessment")

    if report["exercise"] == "Squats":

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Correct Frames",
            report["correct_frames"]
        )

        p2.metric(
            "Incorrect Frames",
            report["incorrect_frames"]
        )

        p3.metric(
            "Unassessed Frames",
            report["unassessed_frames"]
        )

        if report["assessed_frames"] > 0:

            st.metric(
                "Basic Posture Score",
                f'{report["posture_score"]:.1f}%'
            )

            st.caption(
                "This score is the percentage of assessed frames "
                "that passed the implemented squat checks. "
                "It is not a medical or overall fitness score."
            )

        else:

            st.warning(
                "No frames could be assessed. "
                "Check that the person's body is visible."
            )

    else:

        st.info(
            "Pose tracking is available for this exercise, "
            "but its red/green posture rules have not yet "
            "been configured and validated."
        )

    st.subheader("Movement Feedback")

    if report["detection_rate"] < 70:

        st.warning(
            "Low pose visibility. Improve lighting and keep "
            "your body and joints visible."
        )

    elif report["detection_rate"] < 90:

        st.info(
            "Moderate pose visibility. Try a clearer video "
            "with less obstruction."
        )

    else:

        st.success(
            "Pose detected in most frames."
        )

    st.write(
        f'**Final movement:** {report["last_state"]}'
    )

    st.write(
        f'**Pose-visible frames:** '
        f'{report["visible"]} / {report["frames"]}'
    )

    st.subheader("Processed Video")

    st.video(video_bytes)

    st.download_button(
        "Download Assessed Video",
        data=video_bytes,
        file_name="assessed_video.mp4",
        mime="video/mp4"
    )

    st.caption(
        "This is a basic computer-vision demonstration. "
        "Posture checks are approximate and depend on camera "
        "angle, visibility and threshold calibration."
    )
