import cv2
import os

from src.marker_image_detector import detect_markers_frame


def detect_markers_video(
    video_path,
    target_fps=30,
    show_video=True,
    txt_path="outputs/marker_output.txt",
    output_video_path="outputs/marker_analysis_slow.mp4",
    slow_video_fps=10
):

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # Video information
    source_fps = video.get(cv2.CAP_PROP_FPS)

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

    analysis_fps = source_fps / frame_step

    print(f"Original FPS: {source_fps:.2f}")
    print(f"Frame step: {frame_step}")
    print(f"Analysis FPS: {analysis_fps:.2f}")

    # Create output folder
    os.makedirs(
        os.path.dirname(txt_path),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(output_video_path),
        exist_ok=True
    )

    # Create slow-motion output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        slow_video_fps,
        (width, height)
    )

    if not video_writer.isOpened():
        raise RuntimeError(
            "Could not create output video"
        )

    frame_index = 0
    video_results = []

    # Open TXT output file
    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as txt_file:

        # File header
        txt_file.write(
            "MARKER VIDEO ANALYSIS\n"
        )

        txt_file.write(
            "=" * 60 + "\n"
        )

        txt_file.write(
            f"Video: {video_path}\n"
        )

        txt_file.write(
            f"Resolution: {width} x {height}\n"
        )

        txt_file.write(
            f"Original FPS: {source_fps:.2f}\n"
        )

        txt_file.write(
            f"Analysis FPS: {analysis_fps:.2f}\n"
        )

        txt_file.write(
            f"Frame step: {frame_step}\n\n"
        )

        while True:

            success, frame = video.read()

            if not success:
                break

            # Analyze only selected frames
            if frame_index % frame_step == 0:

                markers, marker_mask = (
                    detect_markers_frame(frame)
                )

                timestamp_ms = int(
                    round(
                        frame_index
                        * 1000.0
                        / source_fps
                    )
                )

                # Save results in memory
                video_results.append({
                    "frame": frame_index,
                    "time_ms": timestamp_ms,
                    "markers": markers
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

                txt_file.write(
                    f"Detected markers: "
                    f"{len(markers)}\n"
                )

                for marker_id, marker in enumerate(
                    markers
                ):

                    txt_file.write(
                        f"P{marker_id}: "
                        f"X={marker['x']:.1f} | "
                        f"Y={marker['y']:.1f} | "
                        f"area={marker['area']:.1f}\n"
                    )

                txt_file.write("\n")

                # ---------------------------------
                # DRAW MARKERS ON FRAME
                # ---------------------------------

                for marker_id, marker in enumerate(
                    markers
                ):

                    x = marker["x"]
                    y = marker["y"]

                    # Marker circle
                    cv2.circle(
                        frame,
                        (int(x), int(y)),
                        12,
                        (0, 255, 0),
                        -1
                    )

                    # Marker name
                    cv2.putText(
                        frame,
                        f"P{marker_id}",
                        (int(x) + 15, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                # Add frame number on video
                cv2.putText(
                    frame,
                    f"Frame: {frame_index}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

                # Add timestamp
                cv2.putText(
                    frame,
                    f"Time: {timestamp_ms} ms",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

                # ---------------------------------
                # SAVE SLOW-MOTION VIDEO
                # ---------------------------------

                video_writer.write(frame)

                # ---------------------------------
                # SHOW VIDEO
                # ---------------------------------

                if show_video:

                    display_frame = cv2.resize(
                        frame,
                        None,
                        fx=0.4,
                        fy=0.4
                    )

                    cv2.imshow(
                        "Marker video analysis",
                        display_frame
                    )

                    # 10 FPS preview = 100 ms/frame
                    preview_delay = int(
                        1000 / slow_video_fps
                    )

                    if (
                        cv2.waitKey(preview_delay)
                        & 0xFF
                        == ord("q")
                    ):
                        break

            frame_index += 1

    video.release()
    video_writer.release()

    cv2.destroyAllWindows()

    print(
        f"Marker frames analyzed: "
        f"{len(video_results)}"
    )

    print(
        f"Marker TXT saved: {txt_path}"
    )

    print(
        f"Marker video saved: "
        f"{output_video_path}"
    )

    return video_results