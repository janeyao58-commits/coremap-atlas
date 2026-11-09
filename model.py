"""Apply the existing frozen models; no training or feature selection."""
import json
import math
from pathlib import Path

BUNDLE = json.loads((Path(__file__).parent / 'dist/models.json').read_text('utf-8'))
NAMES = {
    'BN_159_Core': ('躯体感觉2区 A2', 'Somatosensory area 2'),
    'BN_57_Core': ('上肢运动亚区', 'Upper-limb motor area'),
    'JHU_18_Core': ('内囊前肢', 'Anterior limb of internal capsule'),
    'JHU_26_Core': ('上放射冠', 'Superior corona radiata'),
    'JHU_34_Core': ('外囊', 'External capsule'),
    'AAL_14_Core': ('Rolandic盖', 'Rolandic operculum'),
    'AAL_78_Core': ('壳核', 'Putamen'),
    'BN_158_Core': ('舌／喉感觉亚区', 'Tongue/larynx sensory area'),
    'BN_62_Core': ('舌／喉运动亚区', 'Tongue/larynx motor area'),
    'JHU_33_Core': ('外囊', 'External capsule'),
}

def features(side):
    if side not in ('Left', 'Right'):
        raise ValueError('Invalid lesion side')
    return BUNDLE['models'][side + '_M2_Expanded']['Features'][4:]

def predict(side, age, sex, nihss, total, volumes):
    if not all(math.isfinite(float(v)) for v in (age, sex, nihss, total)):
        raise ValueError('Inputs must be finite')
    if not 18 <= age <= 120 or sex not in (0, 1) or not 0 <= nihss <= 42 or total < 0:
        raise ValueError('Invalid clinical inputs or total volume')
    values = dict(Age=age, Sex_binary=sex, NIHSS=nihss, Total_Core_Vol_MNI=total)
    rows = []
    for feature in features(side):
        _, roi, encoding = feature.split('__')
        v = float(volumes[roi])
        if not math.isfinite(v) or v < 0 or v > total + 1e-6:
            raise ValueError('ROI volume must be between 0 and total core: ' + roi)
        value = v if encoding == 'Volume' else float(v > 0) if encoding == 'Binary' else v / total if total > 0 else 0.
        values[feature] = value
        rows.append(dict(ROI=roi, Encoding=encoding, Volume_mL=v, Feature=value))
    probabilities = []
    for role in ('Base', 'Expanded'):
        m = BUNDLE['models'][side + '_M2_' + role]
        z = m['Intercept'] + sum((values[f] - mean) / scale * coef for f, mean, scale, coef in zip(m['Features'], m['Mean'], m['Scale'], m['Coefficients']))
        probabilities.append(1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z)))
    return dict(base=probabilities[0], expanded=probabilities[1], delta_pp=100 * (probabilities[1] - probabilities[0]), rows=rows, model_version=BUNDLE['version'])
