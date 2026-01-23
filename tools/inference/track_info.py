
from dataclasses import dataclass, field
import numpy as np

@dataclass
class TrackInfo:
    track_id: int   # Track ID
    first_box: np.ndarray  # Oldest person detection box  [x1,y1,x2,y2,score]
    last_box: np.ndarray = None # latest person detection box  [x1,y1,x2,y2,score]
    head_box: np.ndarray = None # head box
    face_box: np.ndarray = None # face box
    age: int = -1 # current age
    age_agg: dict = field(default_factory=lambda: {'age_agg_each_bin': [0] * 10, 
                        'num_ages_each_bin': [0] * 10, 
                        'age_pred_hist': [],
                        'age': -1 }) # aggregation age
    gender: int = -1 # current gender
    gender_agg: dict = field(default_factory=lambda: {'gender_agg_list': [0, 0, 0], 
                        'gender': -1}) # aggregation gender
    angles:  list = field(default_factory=list) 
    watching_events: dict = field(default_factory=lambda: {'count_watching_time': False, 'watching_time': 0.0, 'timestamp': None})
    start_time: float = -1# start time of person track
    end_time: float = -1 # end time of person track
    start_frame: int = -1 # start frame of person track
    end_frame: int = -1 # end frame of person track
    dead_time: float = -1 # dead time of person track
    
    last_time_left_hand_in_basket: float = -1 # last time of left hand of person track inside basket
    last_frame_left_hand_in_basket: int = -1 # last frame of left hand of person track inside basket
    last_time_right_hand_in_basket: float = -1 # last time of right hand of person track inside basket
    last_frame_right_hand_in_basket: int = -1 # last frame of right hand of person track inside basket

@dataclass
class ProductTrackInfo:
    track_id: int   # Track ID
    box: np.ndarray  # lastest product detection box  [x1,y1,x2,y2,score,class_id]
    start_time: float = -1 # start time of product track
    end_time: float = -1 # end time of product track
    start_frame: int = -1 # start frame of product track
    end_frame: int = -1 # end frame of product track
    dead_time: float = -1 # dead time of product track
    
    is_hold_by_left_hand: bool = False
    left_hand_kpt: tuple = None
    last_frame_id_hold_by_left_hand: int = -1
    is_hold_by_right_hand: bool = False
    right_hand_kpt: tuple = None
    last_frame_id_hold_by_right_hand: int = -1
    person_id_hold: int = -1
    
    is_start_in_basket: bool = False
    start_time_in_basket: float = -1
    start_frame_in_basket: int = -1
    is_end_in_bag: bool = False
    end_time_in_bag: float = -1
    end_frame_in_bag: int = -1
    
    start_time_scanning: float = -1
    start_frame_scanning: int = -1
    end_time_scanning: float = -1
    end_frame_scanning: int = -1
    is_scanned: bool = False
    is_scanning_by_left_hand: bool = False
    is_scanning_by_right_hand: bool = False
    
@dataclass
class AliveTrack:
    track_id: int   # Person ID
    box: np.ndarray  # latest person detection box  [x1,y1,x2,y2,score]
    kpts: np.ndarray = None # kpt of lastest person detection box [17, 3]
    head_box: np.ndarray = None
    face_box: np.ndarray = None # face box
    age: int = -1 #  age
    gender: int = -1 # gender
    angles:  list = field(default_factory=list)
    is_left_hold_product: bool = False
    is_left_scan_product: bool = False
    left_hand_kpt: tuple = None
    left_action: str = None
    left_hold_product_ids: list = field(default_factory=list)
    is_right_hold_product: bool = False
    is_right_scan_product: bool = False
    right_hand_kpt: tuple = None
    scanning_product_id: int = -1
    right_action: str = None
    right_hold_product_ids: list = field(default_factory=list)
    watching_events: dict = field(default_factory=lambda: {'count_watching_time': False, 'watching_time': 0.0, 'timestamp': None})
    line_passing_events: list = field(default_factory=list)   # List of line passing events
    area_passing_events: list = field(default_factory=list)   # List of area passing events
    
    
@dataclass
class ProductAliveTrack:
    track_id: int   # Product ID
    box: np.ndarray  # latest product detection box  [x1,y1,x2,y2,score]
    is_hold_by_left_hand: bool = False
    left_hand_kpt: tuple = None
    is_hold_by_right_hand: bool = False
    right_hand_kpt: tuple = None
    person_id_hold: int = -1
    is_product_in_basket: bool = False
    is_product_in_bag: bool = False
    is_moving: bool = False
    is_scanned: bool = False
    is_scanning_by_left_hand: bool = False
    is_scanning_by_right_hand: bool = False
    