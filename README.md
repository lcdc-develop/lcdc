# INSTALATION

1. Clone the repository

    ```bash
    git clone https://github.com/kyselica12/lcdc.git
    ```

2. Install the package

    ```bash
    cd lcdc
    pip install .
    ```

# MMT Data



Examples `demo.ipynb`

## Tomas

```python
from datasets import load_dataset, Dataset

from   lcdc import DatasetBuilder, LCDataset
from   lcdc.vars import Variability, TENTH_OF_SECOND, DataType as DT
import lcdc.preprocessing as pp
import lcdc.stats as stats
import lcdc.utils as utils

MMT_PATH = "/home/k/kyselica12/work/mmt/MMT"
classes = ["H-2A R/B"]
regexes = None
preprocessing = pp.Compose(
    pp.FilterByPeriodicity(Variability.PERIODIC),
    pp.SplitByRotationalPeriod(1), 
    pp.FilterFolded(100, 0.75), 
    pp.FilterMinLength(100),
)
statistics = [
    stats.MediumTime(), 
    stats.MediumPhase(), 
    stats.FourierSeries(8, fs=True, amplitude=True)
]

db = DatasetBuilder(MMT_PATH, classes=classes, regexes=regexes, preprocessing=preprocessing, statistics=statistics, lazy=False)
ds = db.to_dict(data_types=[DT.MAG, DT.TIME, DT.PHASE, DT.DISTANCE])

'''
Dataset is in ds["data"]
>>> ds["data"]
Dataset({
    features: ['mag', 'time', 'phase', 'distance', 'id', 'norad_id', 'label', 'period', 'timestamp', 'start_idx', 'end_idx', 'FourierAmplitude', 'FourierCoefs', 'MediumPhase', 'MediumTime'],
    num_rows: 4202
})


''''


```

## TODO

- [ ] Save to parquet??
