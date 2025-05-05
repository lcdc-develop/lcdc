# LCDC: Light curve dataset creator

**LCDC** is a Python package that allows you to work with large light curve datasets in a simple and efficient way. It is designed to be used creation of dataset for machine learning models as it produces output in `datasets.Dataset` format. Also it is a powerful tool for data preprocessing and scientific analysis on whole populations.

It is sutaible for working with [**MMT_snapshot**](https://huggingface.co/datasets/kyselica/MMT_snapshot/) dataset created from MMT database [^1].

For full documentation visit [docs](https://lcdc-develop.github.io/lcdc/).

<!-- For full documentation visit [mkdocs.org](https://www.mkdocs.org). -->

## Instalation

```bash
pip install lcdc
```

## Simple Example

```python
from lcdc import DatasetBuilder
from lcdc import vars
from lcdc import utils
import lcdc.preprocessing as pp
import lcdc.stats as stats

db = DatasetBuilder(DATA_PATH, norad_ids=[IDX])
preprocessing = [
    pp.FilterByPeriodicity(vars.Variability.PERIODIC),
    pp.SplitByRotationalPeriod(1), 
    pp.FilterMinLength(100),
    pp.FilterFolded(100, 0.8), 
]

db.preprocess(preprocessing)
dataset = db.build_dataset()
print(dataset)
```

### Output

```bash
Loaded 402 track
Preprocessing: 100%|██████████| 402/402 [00:08<00:00, 49.28it/s]
Dataset({
    features: ['norad_id', 'id', 'period', 'timestamp', 'time', 'mag', 'phase', 'distance', 'filter', 'name', 'variability', 'label', 'range'],
    num_rows: 4057
})
```

## 📝 Citing

```bibtex
@article{kyselica2025lcdc,
  title={LCDC: Bridging Science and Machine Learning for Light Curve Analysis},
  author={Kyselica, Daniel and Hrob{\'a}r, Tom{\'a}{\v{s}} and {\v{S}}ilha, Ji{\v{r}}{\'\i} and {\v{D}}urikovi{\v{c}}, Roman and {\v{S}}uppa, Marek},
  journal={arXiv preprint arXiv:2504.10550},
  year={2025}
}
```


[^1]: Karpov, S., et al. "Mini-Mega-TORTORA wide-field monitoring system with sub-second temporal resolution: first year of operation." Revista Mexicana de Astronomía y Astrofísica 48 (2016): 91-96.


