import numpy as np
from collections import Counter
from shapely.geometry import Point, Polygon

def cal_dist(point1, point2):
    vector = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    distance = np.linalg.norm(vector)
    return distance

def box_euclid_dist(boxA, boxB):
    xA = (boxA[0] + boxA[2])/ 2
    yA = (boxA[1] + boxA[3])/ 2
    xB = (boxB[0] + boxB[2])/ 2
    yB = (boxB[1] + boxB[3])/ 2
    vector = np.array([xA - xB, yA - yB])
    dist = np.linalg.norm(vector)
    return dist

def is_box_in_roi_min_iou(bbox, rois, min_iou_threshold=0.2):
    for roi in rois:
        score = min_box_iou(bbox, roi)
        if score > min_iou_threshold:
            return True
    return False
    
def get_box_point(bbox, ratio):
    return np.array([bbox[0] + ratio[0] * (bbox[2] - bbox[0]), bbox[1] + ratio[1] * (bbox[3] - bbox[1])], dtype=int)

def is_point_in_rois(point, rois):
    if not point: return False
    for roi in rois:
        x1, y1, x2, y2, _, _ = roi
        if x1 < point[0] < x2 and y1 < point[1] < y2: 
            return True
    return False

def is_box_in_rois(bbox, rois):
    center = (bbox[0] + bbox[2])/ 2, (bbox[1] + bbox[3])/ 2
    for roi in rois:
        x1, y1, x2, y2, _, _ = roi
        if x1 < center[0] < x2 and y1 < center[1] < y2: 
            return True
    return False

def check_product_in_rois(product_box, rois, iou_threshold=0.3, extend_ratio=0.1):
    for roi in rois:
        x1,y1,x2,y2,sc,_ = roi
        x1 -= extend_ratio * (x2 - x1)
        y1 -= extend_ratio * (y2 - y1)
        x2 += extend_ratio *  (x2 - x1)
        y2 += extend_ratio * (y2 - y1)
        min_iou_score = min_box_iou(product_box, [x1, y1, x2, y2])
        if min_iou_score > iou_threshold:
            return True
    return False

def min_box_iou(a, b):
    """
    to calculate ratio between intersection area and smaller box area.
    :param a: array of first bounding box in format [xmin, ymin, xmax, ymax]
    :param b: array of second bounding boxes in format [xmin, ymin, xmax, ymax]
    :return: area of (a and b)/ min( area of a, area of b)
    """
    w_intsec = np.maximum(0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum(0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0]) * (a[3] - a[1])
    s_b = (b[2] - b[0]) * (b[3] - b[1])

    return float(s_intsec) / (np.minimum(s_a, s_b))

def iou(bb_test, bb_gt):
    """
    Computes IUO between two bounding boxes in the form [x1,y1,x2,y2]
    :param bb_test: the first bounding box
    :param bb_gt: the second bounding box
    :return: IOU
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        - wh
    )
    return o

def bbox_center_dist(bbox1, bbox2):
    """calculates distance between centers of two bboxes

    Args:
        bbox1 (tuple): box in format x1, y1, x2, y2
        bbox2 (tuple): box in format x1, y1, x2, y2
    """
    center1 = (bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2
    center2 = (bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(
    	np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
        
    return area_inter / (area_a[:, None] + area_b - area_inter)


def voting(elements):
    c = Counter(elements)
    return c.most_common(1)[0][0]

def regularize(states, ids):
    out = [states[0]]
    out_ids = [ids[0]]
    i = 0
    for idx, state in enumerate(states):
        if state != out[i]:
            out.append(state)
            out_ids.append(ids[idx])
            i += 1
    return out, out_ids

def non_max_suppression(
    predictions: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    rows, columns = predictions.shape

    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    categories = predictions[:, 5]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    return keep[sort_index.argsort()]

def check_single_point_in_area(roi, point):
    if not roi: 
        return True
    if not point:
        return False
    return Polygon(roi).contains(Point(point))

def check_hand_hold_product(hand, product, max_dist_hand2product=100):
    if hand is None: 
        return 10000, False
    x1,y1,x2,y2,score,_ = product
    x,y = hand
    x_satisfied = x1<=x<=x2
    y_satisfied = y1<=y<=y2
    if x_satisfied and y_satisfied:
        return 0, True
    min_dist_x = min(abs(x - x1), abs(x - x2))
    min_dist_y = min(abs(y - y1), abs(y - y2))
    if (x_satisfied and min_dist_y < max_dist_hand2product) or (y_satisfied and min_dist_x < max_dist_hand2product) \
    or (not x_satisfied and not y_satisfied and min_dist_x < max_dist_hand2product and min_dist_y < max_dist_hand2product):
        dist = min(min_dist_x, min_dist_y)
        return dist, True
    return 10000, False

def get_hand_point(keypoints, conf_thres=0.1, scale=0.3):
    if len(keypoints) == 0:
        return (None, None)
    left_elbow = keypoints[7]
    left_wrist = keypoints[9]
    right_elbow = keypoints[8]
    right_wrist = keypoints[10]
    left_hand_kpt = None
    right_hand_kpt = None
    if left_wrist[2] > conf_thres:
        if left_elbow[2] > conf_thres:
            x = left_elbow[0] + (1 + scale) * (left_wrist[0] - left_elbow[0])
            y = left_elbow[1] + (1 + scale) * (left_wrist[1] - left_elbow[1])
            left_hand_kpt = int(x), int(y)
        else:
            left_hand_kpt = int(left_wrist[0]), int(left_wrist[1])
    if right_wrist[2] > conf_thres:
        if right_elbow[2] > conf_thres:
            x = right_elbow[0] + (1 + scale) * (right_wrist[0] - right_elbow[0])
            y = right_elbow[1] + (1 + scale) * (right_wrist[1] - right_elbow[1])
            right_hand_kpt = int(x), int(y)
        else:
            right_hand_kpt = int(right_wrist[0]), int(right_wrist[1])
    return (left_hand_kpt, right_hand_kpt)