"""두 감지 노드가 공유하는 YOLO → ROS 메시지 변환 함수."""

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


def detect(model, frame, header, target_class, confidence, device):
    """클래스 이름이 일치하는 박스만 담는다. 감지가 없으면 빈 배열이다."""
    result = model.predict(frame, conf=confidence, device=device, verbose=False)[0]
    message = Detection2DArray()
    message.header = header
    for box in result.boxes:
        class_id = int(box.cls.item())
        label = str(result.names[class_id])
        if label != target_class:
            continue
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        score = float(box.conf.item())
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score)):
            continue
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
