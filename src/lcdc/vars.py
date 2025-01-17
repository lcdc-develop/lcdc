from enum import IntEnum, StrEnum


NANOSEC = 10 ** 9
TENTH_OF_SECOND =  NANOSEC // 10

class TableCols(StrEnum):
    ID = 'id'
    NORAD_ID = 'norad_id'
    TIMESTAMP = 'timestamp'
    PERIOD = 'period'
    TIME = 'time'
    MAG = 'mag'
    PHASE = 'phase'
    DISTANCE = 'distance'
    FILTER = 'filter'
    VARIABILITY = 'variability'
    NAME = 'name'
    LABEL = 'label'
    RANGE = 'range'

DATA_COLS = [TableCols.TIME, TableCols.MAG, TableCols.PHASE, TableCols.DISTANCE, TableCols.FILTER]

class StrVars(StrEnum):
    FOURIER_COEFS = 'fourier_coefs'
    WAVELET = "wavelet"

class Variability(StrEnum):
    APERIODIC = 'aperiodic'
    PERIODIC = 'periodic'
    NONVARIABLE = 'non-variable'

    @staticmethod
    def from_int(n):
        match n:
            case 0:
                return Variability.NONVARIABLE
            case 1:
                return Variability.APERIODIC
            case 2:
                return Variability.PERIODIC
            case _:
                raise ValueError(f"Unknown variability type: {n}")
                

class DataType(StrEnum):
    TIME = 'time'
    MAG = 'mag'
    PHASE = 'phase'
    DISTANCE = 'distance'
    FILTER = 'filter'

ALL_TYPES = [DataType.TIME, DataType.MAG, DataType.PHASE, DataType.DISTANCE, DataType.FILTER]
TYPES_INDICES = {t: i for i, t in enumerate(ALL_TYPES)}


class Filter(IntEnum):
    UNKNOWN = int('00000',2) # 0
    CLEAR   = int('00001',2) # 1
    POL     = int('00010',2) # 2
    V       = int('00100',2) # 4
    R       = int('01000',2) # 8
    B       = int('10000',2) # 16

    @staticmethod
    def str_to_int(s):
        r = 0
        for n in ["Unknown", "Clear", "Pol", "V", "R", "B"]:
            if n in s:
                r = r | Filter[n.upper()].value
        return r
    
    @staticmethod
    def from_int(n):
        res = set()
        for a in Filter:
            if a & n == a:
                res.add(Filter(a))
        return res
