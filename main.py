from src.mp_image_detector import detect_pose_image

from src.marker_image_detector import detect_markers_image

from src.marker_video_detector import detect_markers_video
from src.mp_video_detector import detect_pose_video

#from src.video_comparison import compare_video_results


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
        target_fps=30,
        show_video=True,
        txt_path="outputs/marker_output.txt",
        output_video_path=(
            "outputs/marker_analysis_slow.mp4"
        ),
        slow_video_fps=10
    )

    print(
        "\n--- MEDIAPIPE ANALYSIS ---\n"
    )

    mediapipe_results = detect_pose_video(
        VIDEO_PATH,
        MODEL_PATH,
        target_fps=30,
        txt_path="outputs/mediapipe_output.txt"
    )

    print(
        "\n--- ANALYSIS COMPLETED ---"
    )

    print(
        f"Marker frames: "
        f"{len(marker_results)}"
    )

    print(
        f"MediaPipe frames: "
        f"{len(mediapipe_results)}"
    )