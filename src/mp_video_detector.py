# Handles MediaPipe pose detection in videos

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LANDMARK_NAMES = {
    0: "naso",

    11: "spalla_sinistra",
    12: "spalla_destra",

    13: "gomito_sinistro",
    14: "gomito_destro",

    15: "polso_sinistro",
    16: "polso_destro",

    23: "anca_sinistra",
    24: "anca_destra",

    25: "ginocchio_sinistro",
    26: "ginocchio_destro",

    27: "caviglia_sinistra",
    28: "caviglia_destra",

    29: "tallone_sinistro",
    30: "tallone_destro",

    31: "avampiede_sinistro",
    32: "avampiede_destro"
}


def detect_pose_video(
    video_path,
    model_path,
    target_fps=30
):

    # Open video
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise FileNotFoundError(
            f"Video not found or cannot be opened: {video_path}"
        )

    # Video information
    source_fps = video.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        video.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(f"Video resolution: {width} x {height}")
    print(f"Original FPS: {source_fps:.2f}")
    print(f"Total frames: {total_frames}")

    # Example:
    # 60 FPS -> 30 FPS = analyze every 2 frames
    frame_step = max(
        1,
        round(source_fps / target_fps)
    )

    analysis_fps = source_fps / frame_step

    print(f"Frame step: {frame_step}")
    print(
        f"Analysis FPS: {analysis_fps:.2f}\n"
    )

    # MediaPipe configuration
    base_options = python.BaseOptions(
        model_asset_path=model_path
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    results = []

    frame_index = 0

    # Create MediaPipe model only once
    with vision.PoseLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = video.read()

            if not success:
                break

            if frame_index % frame_step == 0:

                timestamp_ms = int(
                    round(
                        frame_index
                        * 1000.0
                        / source_fps
                    )
                )

                # OpenCV BGR -> RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # NumPy frame -> MediaPipe image
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # MediaPipe video analysis
                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                landmarks = {}

                if result.pose_landmarks:

                    pose_landmarks = (
                        result.pose_landmarks[0]
                    )

                    for landmark_id, name in LANDMARK_NAMES.items():

                        landmark = pose_landmarks[
                            landmark_id
                        ]

                        x_pixel = landmark.x * width
                        y_pixel = landmark.y * height

                        landmarks[name] = {
                            "x": x_pixel,
                            "y": y_pixel,
                            "visibility": landmark.visibility
                        }

                results.append({
                    "frame": frame_index,
                    "time_ms": timestamp_ms,
                    "landmarks": landmarks
                })

                print(
                    f"Frame {frame_index} | "
                    f"{timestamp_ms} ms | "
                    f"Landmarks detected: "
                    f"{len(landmarks)}"
                )

            frame_index += 1

    video.release()

    return results