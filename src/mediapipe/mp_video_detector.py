# Questo modulo analizza un video con MediaPipe.
# Legge i frame del video, rileva i landmark corporei per ciascun frame
# analizzato e restituisce i risultati insieme alle informazioni del video.

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

    video = cv2.VideoCapture(
        video_path
    )

    if not video.isOpened():

        raise FileNotFoundError(
            f"Video not found: "
            f"{video_path}"
        )

    source_fps = video.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        video.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        video.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    frame_step = max(
        1,
        round(
            source_fps / target_fps
        )
    )

    analysis_fps = (
        source_fps / frame_step
    )

    base_options = (
        python.BaseOptions(
            model_asset_path=model_path
        )
    )

    options = (
        vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=(
                vision.RunningMode.VIDEO
            ),
            num_poses=1
        )
    )

    frames = []

    frame_index = 0

    with (
        vision.PoseLandmarker
        .create_from_options(
            options
        )
    ) as landmarker:

        while True:

            success, frame = (
                video.read()
            )

            if not success:
                break

            if (
                frame_index
                % frame_step
                == 0
            ):

                timestamp_ms = int(
                    round(
                        frame_index
                        * 1000.0
                        / source_fps
                    )
                )

                rgb_frame = (
                    cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )
                )

                mp_image = mp.Image(
                    image_format=(
                        mp.ImageFormat.SRGB
                    ),
                    data=rgb_frame
                )

                result = (
                    landmarker
                    .detect_for_video(
                        mp_image,
                        timestamp_ms
                    )
                )

                landmarks = {}

                if result.pose_landmarks:

                    pose = (
                        result
                        .pose_landmarks[0]
                    )

                    for (
                        landmark_id,
                        name
                    ) in (
                        LANDMARK_NAMES
                        .items()
                    ):

                        landmark = (
                            pose[
                                landmark_id
                            ]
                        )

                        landmarks[name] = {

                            "x": (
                                landmark.x
                                * width
                            ),

                            "y": (
                                landmark.y
                                * height
                            ),

                            "z": landmark.z,

                            "visibility":
                                landmark.visibility
                        }

                frames.append({

                    "frame":
                        frame_index,

                    "time_ms":
                        timestamp_ms,

                    "landmarks":
                        landmarks
                })

            frame_index += 1

    video.release()

    return {
        "video_path": video_path,

        "width": width,
        "height": height,

        "source_fps": source_fps,
        "analysis_fps": analysis_fps,

        "frame_step": frame_step,
        "total_frames": total_frames,

        "frames": frames
    }