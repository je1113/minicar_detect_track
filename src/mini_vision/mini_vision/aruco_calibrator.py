import cv2
import numpy as np


def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_250
    )

    parameters = cv2.aruco.DetectorParameters_create()

    cap = cv2.VideoCapture(4, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError('웹캠을 열 수 없습니다.')

    while True:
        ok, frame = cap.read()

        if not ok:
            continue

        corners, ids, rejected = cv2.aruco.detectMarkers(
            frame,
            aruco_dict,
            parameters=parameters
        )

        if ids is not None:
            ids = ids.flatten()

            for marker_corners, marker_id in zip(corners, ids):
                pts = marker_corners[0]

                center_x = float(np.mean(pts[:, 0]))
                center_y = float(np.mean(pts[:, 1]))

                cv2.polylines(
                    frame,
                    [pts.astype(np.int32)],
                    True,
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (int(center_x), int(center_y)),
                    5,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    f'ID {marker_id} ({center_x:.1f}, {center_y:.1f})',
                    (int(center_x) + 8, int(center_y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                print(
                    f'ID {marker_id}: '
                    f'center=({center_x:.1f}, {center_y:.1f})'
                )

        cv2.imshow('ArUco Calibration', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()