from src.mediapipe.mp_image_detector import detect_pose_image
from src.validation.marker_image_detector import detect_markers_image

from src.validation.marker_video_detector import detect_markers_video
from src.mediapipe.mp_video_detector import detect_pose_video

from src.validation.video_comparison import create_validation_outputs


MODE = "comparison_video"

MODEL_PATH = "models/pose_landmarker_full.task"

IMAGE_PATH = "images/test_9.jpeg"

VIDEO_PATH = "videos/marker_squat_2.mp4"


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


elif MODE == "comparison_video":

    print(
        "\n--- MARKER ANALYSIS ---\n"
    )

    marker_results = detect_markers_video(
        VIDEO_PATH,
        target_fps=30
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
        "\n--- CREATE VALIDATION OUTPUTS ---\n"
    )

    create_validation_outputs(
        marker_results,
        mediapipe_results,
        output_dir="outputs/validation",
        slow_video_fps=10
    )

    print(
        "\n--- ANALYSIS COMPLETED ---"
    )

    print(
        f"Marker frames: "
        f"{len(marker_results['frames'])}"
    )

    print(
        f"MediaPipe frames: "
        f"{len(mediapipe_results['frames'])}"
    )