import cv2
import mediapipe as mp
import os

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
    target_fps=30,
    txt_path="outputs/mediapipe_output.txt"
):

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise FileNotFoundError(
            f"Video not found or cannot be opened: "
            f"{video_path}"
        )

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

    frame_step = max(
        1,
        round(source_fps / target_fps)
    )

    analysis_fps = (
        source_fps / frame_step
    )

    print(
        f"Video resolution: "
        f"{width} x {height}"
    )

    print(
        f"Original FPS: "
        f"{source_fps:.2f}"
    )

    print(
        f"Total frames: "
        f"{total_frames}"
    )

    print(
        f"Frame step: "
        f"{frame_step}"
    )

    print(
        f"Analysis FPS: "
        f"{analysis_fps:.2f}"
    )

    # Create output directory
    os.makedirs(
        os.path.dirname(txt_path),
        exist_ok=True
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

    # Open TXT file
    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as txt_file:

        # File header
        txt_file.write(
            "MEDIAPIPE VIDEO ANALYSIS\n"
        )

        txt_file.write(
            "=" * 60 + "\n"
        )

        txt_file.write(
            f"Video: {video_path}\n"
        )

        txt_file.write(
            f"Resolution: "
            f"{width} x {height}\n"
        )

        txt_file.write(
            f"Original FPS: "
            f"{source_fps:.2f}\n"
        )

        txt_file.write(
            f"Analysis FPS: "
            f"{analysis_fps:.2f}\n"
        )

        txt_file.write(
            f"Frame step: "
            f"{frame_step}\n\n"
        )

        with vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:

                success, frame = video.read()

                if not success:
                    break

                # Analyze exactly the same frames
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

                    # NumPy -> MediaPipe image
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame
                    )

                    result = (
                        landmarker.detect_for_video(
                            mp_image,
                            timestamp_ms
                        )
                    )

                    landmarks = {}

                    if result.pose_landmarks:

                        pose_landmarks = (
                            result.pose_landmarks[0]
                        )

                        for (
                            landmark_id,
                            name
                        ) in LANDMARK_NAMES.items():

                            landmark = (
                                pose_landmarks[
                                    landmark_id
                                ]
                            )

                            x_pixel = (
                                landmark.x
                                * width
                            )

                            y_pixel = (
                                landmark.y
                                * height
                            )

                            landmarks[name] = {
                                "x": x_pixel,
                                "y": y_pixel,
                                "visibility":
                                    landmark.visibility
                            }

                    # Save in memory
                    results.append({
                        "frame": frame_index,
                        "time_ms": timestamp_ms,
                        "landmarks": landmarks
                    })

                    # ---------------------------------
                    # WRITE RESULTS TO TXT
                    # ---------------------------------

                    txt_file.write(
                        "=" * 60 + "\n"
                    )

                    txt_file.write(
                        f"FRAME {frame_index} | "
                        f"{timestamp_ms} ms\n\n"
                    )

                    if landmarks:

                        for (
                            name,
                            landmark
                        ) in landmarks.items():

                            txt_file.write(
                                f"{name}: "
                                f"X={landmark['x']:.1f} | "
                                f"Y={landmark['y']:.1f} | "
                                f"visibility="
                                f"{landmark['visibility']:.3f}\n"
                            )

                    else:

                        txt_file.write(
                            "No pose detected\n"
                        )

                    txt_file.write("\n")

                frame_index += 1

    video.release()

    print(
        f"MediaPipe frames analyzed: "
        f"{len(results)}"
    )

    print(
        f"MediaPipe TXT saved: "
        f"{txt_path}"
    )

    return results