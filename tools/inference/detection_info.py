from dataclasses import dataclass
import numpy as np

@dataclass
class DetectionInfo:
    person_box: np.ndarray = None
    head_box: np.ndarray = None 
    face_box: np.ndarray = None 
    person_kpts: np.ndarray = None
    
@dataclass
class ProductDetectionInfo:
    box: np.ndarray = None
    is_hold_by_left_hand: bool = False
    left_hand_kpt: tuple = None
    is_hold_by_right_hand: bool = False
    right_hand_kpt: tuple = None
    person_id_hold: int = -1
    is_product_in_basket: bool = False
    is_product_in_bag: bool = False