import cv2

from src.marker_image_detector import detect_markers_frame


DISPLAY_SCALE = 1


def detect_markers_video(
    video_path,
    initial_marker_map,
    target_fps=30,
    show_video=True
):

    # Open video
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise FileNotFoundError(
            f"Video not found or cannot be opened: {video_path}"
        )

    # Read video information
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
    # 60 FPS / 30 FPS = analyze one frame every 2 frames
    frame_step = max(
        1,
        round(source_fps / target_fps)
    )

    analysis_fps = source_fps / frame_step

    print(f"Frame step: {frame_step}")
    print(f"Analysis FPS: {analysis_fps:.2f}\n")

    results = []

    frame_index = 0

    while True:

        success, frame = video.read()

        # End of video
        if not success:
            break

        # Analyze only selected frames
        if frame_index % frame_step == 0:

            # Timestamp relative to the ORIGINAL video
            timestamp_ms = (
                frame_index * 1000.0 / source_fps
            )

            # Analyze this single frame
            markers, marker_mask = detect_markers_frame(
                frame
            )

            results.append({
                "frame": frame_index,
                "time_ms": timestamp_ms,
                "markers": markers
            })

            print(
                f"Frame {frame_index} | "
                f"{timestamp_ms:.1f} ms | "
                f"Markers detected: {len(markers)}"
            )

            if show_video:

                debug_frame = frame.copy()

                for marker_id, marker in enumerate(markers):

                    x = int(marker["x"])
                    y = int(marker["y"])

                    cv2.circle(
                        debug_frame,
                        (x, y),
                        12,
                        (0, 255, 0),
                        3
                    )

                    cv2.putText(
                        debug_frame,
                        str(marker_id),
                        (x + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                cv2.putText(
                    debug_frame,
                    f"Frame: {frame_index}",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3
                )

                display_frame = cv2.resize(
                    debug_frame,
                    None,
                    fx=DISPLAY_SCALE,
                    fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_AREA
                )

                cv2.imshow(
                    "Marker video analysis",
                    display_frame
                )

                # Stop on the first analyzed frame
                if frame_index == 0:

                    print(
                        "\nFirst frame paused."
                        "\nLook at the marker IDs."
                        "\nPress any key to continue.\n"
                    )

                    cv2.waitKey(0)

                else:

                    # Slow playback
                    if cv2.waitKey(100) & 0xFF == ord("q"):
                        break

        frame_index += 1

    if show_video:

        print(
            "\nVideo finished."
            "\nPress any key to close the window."
        )

        cv2.waitKey(0)


    video.release()
    cv2.destroyAllWindows()

    return results

    return results