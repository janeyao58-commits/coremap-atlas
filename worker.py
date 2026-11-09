"""One isolated process per imaging job; exit releases ANTs memory."""
import os
for name in ('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'NSLOTS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['ANTS_RANDOM_SEED'] = '20260921'
from pathlib import Path
import json
import sys
import gzip
import shutil

def validate(path):
    import nibabel as nib
    import numpy as np
    im = nib.load(str(path))
    if len(im.shape) != 3 or any(d <= 0 or d > 1024 for d in im.shape) or np.prod(im.shape) > 32_000_000:
        raise ValueError('Only 3D images up to 32 million voxels are supported.')
    if im.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('Spatial units must be mm.')

def run(folder, mode, side):
    for name in (('cbf', 'core') if mode == 'native' else ('core',)):
        raw = folder / (name + '.upload')
        target = folder / (name + '.nii.gz')
        with raw.open('rb') as stream:
            compressed = stream.read(2) == b'\x1f\x8b'
        if compressed:
            raw.rename(target)
        else:
            with raw.open('rb') as stream, gzip.open(target, 'wb') as output:
                shutil.copyfileobj(stream, output, 1024 * 1024)
            raw.unlink()
        validate(target)
    if mode == 'native':
        from registration import register
        register(folder/'cbf.nii.gz', folder/'core.nii.gz', folder, side, progress=lambda text: print(text, flush=True))
    elif mode == 'mni':
        from predictor import quantify
        result = quantify(folder/'core.nii.gz', side)
        shutil.copyfile(folder/'core.nii.gz', folder/'Core_MNI.nii.gz')
        (folder/'result.json').write_text(json.dumps(result), 'utf-8')
    else:
        raise ValueError('Invalid image mode')

if __name__ == '__main__':
    folder = Path(sys.argv[1]).resolve()
    try:
        run(folder, sys.argv[2], sys.argv[3])
    except Exception as error:
        (folder/'error.json').write_text(json.dumps({'error': str(error)}), 'utf-8')
        raise
