"""두 감지 노드가 공유하는 YOLO → ROS 메시지 변환 함수."""

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


def detect(model, frame, header, target_class, confidence, device):
    """
    target_class와 일치하고 confidence 이상인 객체 중
    가장 confidence가 높은 1개만 ROS Detection2DArray로 반환한다.

    감지가 없으면 빈 Detection2DArray를 반환한다.
    """

    # YOLO 추론
    result = model.predict(
        frame,
        conf=confidence,
        device=device,
        verbose=False,
    )[0]

    message = Detection2DArray()
    message.header = header

    # target_class에 해당하는 후보만 저장
    candidates = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        label = str(result.names[class_id])

        # car가 아니면 무시
        if label != target_class:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        score = float(box.conf.item())

        # NaN / inf 방지
        if not all(
            math.isfinite(v)
            for v in (x1, y1, x2, y2, score)
        ):
            continue

        candidates.append(
            {
                'score': score,
                'label': label,
                'bbox': (x1, y1, x2, y2),
            }
        )

    # car가 하나도 없으면 빈 배열 반환
    if not candidates:
        return message, result

    # confidence가 가장 높은 car 하나 선택
    best = max(candidates, key=lambda item: item['score'])

    x1, y1, x2, y2 = best['bbox']
    score = best['score']
    label = best['label']

    detection = Detection2D()
    detection.header = header

    detection.bbox.center.position.x = (x1 + x2) / 2.0
    detection.bbox.center.position.y = (y1 + y2) / 2.0
    detection.bbox.size_x = x2 - x1
    detection.bbox.size_y = y2 - y1

    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = label
    hypothesis.hypothesis.score = score

    detection.results.append(hypothesis)

    # 가장 confidence 높은 car 하나만 추가
    message.detections.append(detection)

    return message, result


def run_node(node_class, args=None):
    """main() 공통 처리. 카메라 등은 각 노드의 destroy_node()에서 해제한다."""

    rclpy.init(args=args)
    node = None

    try:
        node = node_class()
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()