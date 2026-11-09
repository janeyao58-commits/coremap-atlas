"""Research-grid atlas quantification for the unchanged frozen models."""
from research_processing import read_rpi, load_atlas, measure_regions
from pathlib import Path
from functools import lru_cache
import json
import os
import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parent
TEMPLATES = Path(os.environ.get('COREMAP_TEMPLATES', str(ROOT / 'templates')))


def find(name):
    hits = list(TEMPLATES.rglob(name))
    if len(hits) != 1:
        raise ValueError(f'Template not uniquely available: {name}')
    return hits[0]


@lru_cache(maxsize=1)
def resources():
    reference = read_rpi(find('MNI152_T1_1mm_brain.nii.gz'))
    atlases = {}
    for key, name in [('AAL', 'AAL3v1_1mm.nii'), ('BN', 'BN_Atlas_246_1mm.nii.gz'), ('JHU', 'JHU-ICBM-labels-1mm.nii.gz')]:
        atlases[key] = load_atlas(find(name), reference)
    return reference, atlases


def quantify(path, side):
    import ants
    if side not in ('Left', 'Right'):
        raise ValueError('Select Left or Right.')
    image = nib.load(str(path))
    if len(image.shape) != 3 or image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('NIfTI must be 3D with mm spatial units.')
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or not np.isin(data, [0, 1]).all():
        raise ValueError('A finite binary 0/1 core mask is required.')
    del data
    reference, atlases = resources()
    mask = read_rpi(path)
    if mask.shape != reference.shape or not ants.image_physical_space_consistency(reference, mask):
        raise ValueError('Mask grid differs from the configured MNI152 1-mm template.')
    values = measure_regions(mask, reference, atlases)
    model = json.loads((ROOT / 'dist/models.json').read_text(encoding='utf-8'))['models'][side+'_M2_Expanded']
    volumes = {}
    for feature in model['Features'][4:]:
        roi = feature.split('__')[1]
        volumes[roi] = values.get(roi, 0.0)
    return {'total': values['Total_Core_Vol'], 'volumes': volumes, 'space': 'MNI152 1mm'}
