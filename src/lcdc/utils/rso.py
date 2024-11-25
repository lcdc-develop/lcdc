from dataclasses import dataclass

from ..vars import Variability

@dataclass
class RSO:
    mmt_id : int
    norad_id : int
    name : str
    country : str
    variability : Variability