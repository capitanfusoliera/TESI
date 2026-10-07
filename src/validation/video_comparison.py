import cv2
import os


def create_validation_outputs(
    marker_results,
    mediapipe_results,
    output_dir="outputs/validation",
    slow_video_fps=10
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    marker_txt_path = os.path.join(
        output_dir,
        "marker_output.txt"
    )

    mediapipe_txt_path = os.path.join(
        output_dir,
        "mediapipe_output.txt"
    )

    marker_video_path = os.path.join(
        output_dir,
        "marker_analysis_slow.mp4"
    )

    save_marker_txt(
        marker_results,
        marker_txt_path
    )

    save_mediapipe_txt(
        mediapipe_results,
        mediapipe_txt_path
    )

    save_marker_video(
        marker_results,
        marker_video_path,
        slow_video_fps
    )


def save_marker_txt(
    results,
    output_path
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "MARKER VIDEO ANALYSIS\n"
        )

        file.write(
            "=" * 60 + "\n"
        )

        file.write(
            f"Video: "
            f"{results['video_path']}\n"
        )

        file.write(
            f"Resolution: "
            f"{results['width']} x "
            f"{results['height']}\n"
        )

        file.write(
            f"Original FPS: "
            f"{results['source_fps']:.2f}\n"
        )

        file.write(
            f"Analysis FPS: "
            f"{results['analysis_fps']:.2f}\n"
        )

        file.write(
            f"Frame step: "
            f"{results['frame_step']}\n\n"
        )

        for frame_data in (
            results["frames"]
        ):

            frame = (
                frame_data["frame"]
            )

            time_ms = (
                frame_data["time_ms"]
            )

            markers = (
                frame_data["markers"]
            )

            file.write(
                "=" * 60 + "\n"
            )

            file.write(
                f"FRAME {frame} | "
                f"{time_ms} ms\n\n"
            )

            file.write(
                f"Detected markers: "
                f"{len(markers)}\n"
            )

            for (
                marker_id,
                marker
            ) in enumerate(markers):

                file.write(
                    f"P{marker_id}: "
                    f"X={marker['x']:.1f} | "
                    f"Y={marker['y']:.1f} | "
                    f"area={marker['area']:.1f}\n"
                )

            file.write("\n")


def save_mediapipe_txt(
    results,
    output_path
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "MEDIAPIPE VIDEO ANALYSIS\n"
        )

        file.write(
            "=" * 60 + "\n"
        )

        file.write(
            f"Video: "
            f"{results['video_path']}\n"
        )

        file.write(
            f"Resolution: "
            f"{results['width']} x "
            f"{results['height']}\n"
        )

        file.write(
            f"Original FPS: "
            f"{results['source_fps']:.2f}\n"
        )

        file.write(
            f"Analysis FPS: "
            f"{results['analysis_fps']:.2f}\n"
        )

        file.write(
            f"Frame step: "
            f"{results['frame_step']}\n\n"
        )

        for frame_data in (
            results["frames"]
        ):

            frame = (
                frame_data["frame"]
            )

            time_ms = (
                frame_data["time_ms"]
            )

            landmarks = (
                frame_data["landmarks"]
            )

            file.write(
                "=" * 60 + "\n"
            )

            file.write(
                f"FRAME {frame} | "
                f"{time_ms} ms\n\n"
            )

            if not landmarks:

                file.write(
                    "No pose detected\n\n"
                )

                continue

            for (
                name,
                landmark
            ) in landmarks.items():

                file.write(
                    f"{name}: "
                    f"X={landmark['x']:.1f} | "
                    f"Y={landmark['y']:.1f} | "
                    f"Z={landmark['z']:.4f} | "
                    f"visibility="
                    f"{landmark['visibility']:.3f}\n"
                )

            file.write("\n")


def save_marker_video(
    results,
    output_path,
    slow_video_fps=10
):

    video_path = (
        results["video_path"]
    )

    video = cv2.VideoCapture(
        video_path
    )

    if not video.isOpened():

        raise FileNotFoundError(
            f"Video not found: "
            f"{video_path}"
        )

    width = results["width"]
    height = results["height"]

    frame_results = {

        frame_data["frame"]:
            frame_data

        for frame_data
        in results["frames"]
    }

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        slow_video_fps,
        (width, height)
    )

    frame_index = 0

    while True:

        success, frame = (
            video.read()
        )

        if not success:
            break

        if frame_index in frame_results:

            frame_data = (
                frame_results[
                    frame_index
                ]
            )

            markers = (
                frame_data["markers"]
            )

            timestamp_ms = (
                frame_data["time_ms"]
            )

            for (
                marker_id,
                marker
            ) in enumerate(markers):

                x = int(marker["x"])
                y = int(marker["y"])

                cv2.circle(
                    frame,
                    (x, y),
                    12,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    f"P{marker_id}",
                    (x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                frame,
                f"Frame: {frame_index}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Time: {timestamp_ms} ms",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            writer.write(
                frame
            )

        frame_index += 1

    video.release()
    writer.release()