from dataclasses import dataclass, field

import numpy as np
TRACK_DATA_DIM = 5

@dataclass
class Track:
    id: int
    norad_id: int
    timestamp: str
    mjd: float
    period: int
    start_idx: int = 0
    end_idx: int = -1
    stats: dict = field(default_factory=dict)
    data: np.ndarray = None

    
    def load_data_from_file(self, dir):
        data = np.loadtxt(f"{dir}/{self.norad_id}/{self.id}.csv", delimiter=',', skiprows=1)
        data = np.atleast_2d(data)
        self.data = data[self.start_idx:None if self.end_idx == -1 else (self.end_idx + 1)]
    
    def load_data(self, data):
        self.data = data[self.start_idx:None if self.end_idx == -1 else (self.end_idx + 1)].copy()

    def unload_data(self):
        del self.data
        self.data = None    