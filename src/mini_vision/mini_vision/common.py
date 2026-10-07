"""두 감지 노드가 공유하는 YOLO → ROS 메시지 변환 함수."""

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


def detect(model, frame, header, target_classes, confidence, device):
    """요청한 클래스마다 신뢰도가 가장 높은 객체 1개씩 반환한다.

    target_class='car': car 최대 1개.
    target_class=['car', 'dummy']: car 최대 1개, dummy 최대 1개.
    기존 AMR 코드와의 호환을 위해 인자 이름 target_class를 유지한다.
    """
    if isinstance(target_classes, str):
        target_classes = [target_classes]
    else:
        target_classes = list(target_classes)
    target_classes = list(dict.fromkeys(target_classes))

    result = model.predict(
        frame, conf=confidence, device=device, verbose=False
    )[0]

    message = Detection2DArray()
    message.header = header
    if result.boxes is None:
        return message, result

    best_by_class = {}
    for box in result.boxes:
        class_id = int(box.cls.item())
        label = str(result.names[class_id])
        if label not in target_classes:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        score = float(box.conf.item())
        # 유효하지 않은 숫자와 크기가 없는 박스는 제외한다.
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue

        previous = best_by_class.get(label)
        if previous is None or score > previous['score']:
            best_by_class[label] = {
                'score': score,
                'bbox': (x1, y1, x2, y2),
            }

    for label in target_classes:
        best = best_by_class.get(label)
        if best is None:
            continue
        x1, y1, x2, y2 = best['bbox']

        detection = Detection2D()
        detection.header = header
        # 바운딩 박스 중심과 크기는 픽셀 단위다.
        detection.bbox.center.position.x = (x1 + x2) / 2.0
        detection.bbox.center.position.y = (y1 + y2) / 2.0
        detection.bbox.size_x = x2 - x1
        detection.bbox.size_y = y2 - y1

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = label
        hypothesis.hypothesis.score = best['score']
        detection.results.append(hypothesis)
        message.detections.append(detection)

    return message, result


def run_node(node_class, args=None):
    """노드 실행과 종료 처리. 카메라는 각 노드에서 해제한다."""
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
