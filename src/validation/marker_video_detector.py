import cv2

from src.validation.marker_image_detector import detect_markers_frame

def detect_markers_video(
    video_path,
    target_fps=30
):

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
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

    frame_index = 0
    frames = []

    while True:

        success, frame = video.read()

        if not success:
            break

        if frame_index % frame_step == 0:

            markers, _ = (
                detect_markers_frame(
                    frame
                )
            )

            timestamp_ms = int(
                round(
                    frame_index
                    * 1000.0
                    / source_fps
                )
            )

            frames.append({
                "frame": frame_index,
                "time_ms": timestamp_ms,
                "markers": markers
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