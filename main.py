from src.mp_image_detector import detect_pose_image
from src.marker_image_detector import detect_markers_image

from src.marker_video_detector import detect_markers_video
from src.mp_video_detector import detect_pose_video

from src.video_comparison import compare_video_results


MODE = "comparison_video"

INITIAL_MARKER_MAP = {
    0: "caviglia_destra",
    1: "ginocchio_destro",
    2: "anca_destra",
    5: "spalla_destra"
}

MODEL_PATH = "models/pose_landmarker_full.task"

IMAGE_PATH = "images/test_9.jpeg"

VIDEO_PATH = "videos/marker_squat_2.mp4"


INITIAL_MARKER_MAP = {
    # ESEMPIO:
    # 0: "ginocchio_sinistro",
    # 1: "anca_sinistra",
    # 2: "caviglia_sinistra",
}


if MODE == "image":

    landmarks = detect_pose_image(
        IMAGE_PATH,
        MODEL_PATH
    )

    markers = detect_markers_image(
        IMAGE_PATH
    )

    print(
        f"\nDetected markers: {len(markers)}"
    )

    print(
        f"Detected landmarks: {len(landmarks)}"
    )


elif MODE == "marker_video":

    marker_results = detect_markers_video(
        VIDEO_PATH,
        INITIAL_MARKER_MAP,
        target_fps=30,
        show_video=True
    )

    print(
        f"\nMarker frames analyzed: "
        f"{len(marker_results)}"
    )


elif MODE == "comparison_video":

    print(
        "\n--- MARKER ANALYSIS ---\n"
    )

    marker_results = detect_markers_video(
        VIDEO_PATH,
        INITIAL_MARKER_MAP,
        target_fps=30,
        show_video=True
    )

    print(
        "\n--- MEDIAPIPE ANALYSIS ---\n"
    )

    mediapipe_results = detect_pose_video(
        VIDEO_PATH,
        MODEL_PATH,
        target_fps=30
    )

    print(
        f"\nMarker frames: "
        f"{len(marker_results)}"
    )

    print(
        f"MediaPipe frames: "
        f"{len(mediapipe_results)}"
    )

    if len(marker_results) != len(mediapipe_results):

        print(
            "WARNING: different number "
            "of analyzed frames"
        )

    comparisons = compare_video_results(
        marker_results,
        mediapipe_results,
        csv_path="comparison_results.csv"
    )

    print(
        f"\nValid comparisons: "
        f"{len(comparisons)}"
    )