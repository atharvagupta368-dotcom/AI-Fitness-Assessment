```
# AI-Based Fitness & Posture Assessment System
```

```
An educational computer-vision project that uses MediaPipe pose landmarks to
count squats and provide basic movement feedback.
```

# `## Features` 

- `Human pose landmark detection` 

- `Squat repetition counting` 

- `Knee-angle and torso-lean estimates` 

- `Rule-based posture feedback` 

- `Prerecorded video analysis` 

- `Experimental live-camera dashboard` 

```
## Technologies
- Python
- OpenCV
- MediaPipe
- NumPy
- Streamlit
```

- `Streamlit WebRTC` 

# `## Installation` 

```
Open CMD inside the project folder:
```

```
    .venv\Scripts\activate
    python -m pip install -r requirements.txt
```

```
## Run the dashboard
```

```
    python -m streamlit run app.py
```

```
## Run prerecorded video analysis
```

```
    python main.py
```

```
## Required files
- main.py
- app.py
- pose_landmarker_lite.task
- squat.mp4 (for the sample-video workflow)
- requirements.txt
```

# `## Testing` 

```
Test repetition counting, visibility, different camera angles, lighting
conditions, and live-camera access. Record actual results before making accuracy
claims.
```

# `## Limitations` 

```
This is an educational prototype, not a clinically validated posture assessor.
Its angle thresholds and feedback rules are experimental and may produce
incorrect warnings.
```

