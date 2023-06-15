import numpy as np
import pandas as pd
# pre process
dtypes={
    'elapsed_time':np.int32,
    'event_name':'category',
    'name':'category',
    'level':np.uint8,
    'room_coor_x':np.float32,
    'room_coor_y':np.float32,
    'screen_coor_x':np.float32,
    'screen_coor_y':np.float32,
    'hover_duration':np.float32,
    'text':'category',
    'fqid':'category',
    'room_fqid':'category',
    'text_fqid':'category',
    'fullscreen':'category',
    'hq':'category',
    'music':'category',
    'level_group':'category'}
train_raw = pd.read_csv("./input/predict-student-performance-from-game-play/train.csv", dtype=dtypes)

buf = train_raw[train_raw['level_group'] == "0-4"].reset_index(drop=True)
buf.to_csv("./input/predict-student-performance-from-game-play/train0-4.csv", index=False)
buf = train_raw[train_raw['level_group'] == "5-12"].reset_index(drop=True)
buf.to_csv("./input/predict-student-performance-from-game-play/train5-12.csv", index=False)
buf = train_raw[train_raw['level_group'] == "13-22"].reset_index(drop=True)
buf.to_csv("./input/predict-student-performance-from-game-play/train13-22.csv", index=False)