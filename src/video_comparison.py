import math
import csv


def compare_video_results(
    marker_results,
    mediapipe_results,
    csv_path="comparison_results.csv"
):

    comparisons = []

    for marker_frame, mp_frame in zip(
        marker_results,
        mediapipe_results
    ):

        # Safety check
        if marker_frame["frame"] != mp_frame["frame"]:

            print(
                "Synchronization error: "
                f"{marker_frame['frame']} != "
                f"{mp_frame['frame']}"
            )

            continue

        frame_index = marker_frame["frame"]

        timestamp_ms = marker_frame["time_ms"]

        markers = marker_frame["markers"]

        landmarks = mp_frame["landmarks"]

        for joint_name, marker in markers.items():

            # Marker missing in this frame
            if marker is None:
                continue

            # MediaPipe landmark missing
            if joint_name not in landmarks:
                continue

            mp_landmark = landmarks[joint_name]

            marker_x = marker["x"]
            marker_y = marker["y"]

            mp_x = mp_landmark["x"]
            mp_y = mp_landmark["y"]

            error_x = marker_x - mp_x
            error_y = marker_y - mp_y

            error_distance = math.sqrt(
                error_x**2 +
                error_y**2
            )

            comparison = {
                "frame": frame_index,
                "time_ms": timestamp_ms,
                "joint": joint_name,

                "marker_x": marker_x,
                "marker_y": marker_y,

                "mediapipe_x": mp_x,
                "mediapipe_y": mp_y,

                "error_x": error_x,
                "error_y": error_y,

                "error_px": error_distance,

                "visibility": mp_landmark[
                    "visibility"
                ]
            }

            comparisons.append(
                comparison
            )

            print(
                f"Frame {frame_index} | "
                f"{joint_name} | "
                f"Marker=({marker_x:.1f}, {marker_y:.1f}) | "
                f"MP=({mp_x:.1f}, {mp_y:.1f}) | "
                f"Error={error_distance:.2f} px"
            )

    save_comparison_csv(
        comparisons,
        csv_path
    )

    return comparisons


def save_comparison_csv(
    comparisons,
    csv_path
):

    if not comparisons:
        print("No valid comparisons")
        return

    fieldnames = [
        "frame",
        "time_ms",
        "joint",
        "marker_x",
        "marker_y",
        "mediapipe_x",
        "mediapipe_y",
        "error_x",
        "error_y",
        "error_px",
        "visibility"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            comparisons
        )

    print(
        f"\nComparison saved to: {csv_path}"
    )