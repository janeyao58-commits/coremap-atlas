"""Processing adapted from the user-confirmed GLZ_166_Rerun_ROI.py.

Only job-local output paths and error handling are adapted for the web service.
Registration uses research defaults with a fixed seed and one ITK thread.
"""
import os

for name in ('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'NSLOTS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['ANTS_RANDOM_SEED'] = '20260921'

from pathlib import Path
import numpy as np


def read_rpi(path):
    import ants
    image = ants.image_read(str(path))
    if image.orientation != 'RPI':
        image = ants.reorient_image2(image, 'RPI')
    return image


def load_atlas(path, reference):
    import ants
    image = ants.image_read(str(path))
    if not ants.image_physical_space_consistency(reference, image):
        image = ants.resample_image_to_target(image, reference, interp_type='genericLabel')
    return image


def register_core(cbf_path, core_path, template_path, output, progress):
    import ants
    # ANTsPy 0.6.3 reads its seed from config, not the legacy keyword.
    # Keep Rigid/SyN unchanged; deterministic=True would require Repro transforms.
    ants.config.set_ants_deterministic(on=False, seed_value=20260921)
    output = Path(output)
    reference = read_rpi(template_path)
    moving = read_rpi(cbf_path)
    progress('Registering CBF: research Rigid initialization…')
    rigid = ants.registration(fixed=reference, moving=moving,
                              type_of_transform='Rigid',
                              outprefix=str(output / 'rigid_'))
    progress('Registering CBF: research SyN after Rigid…')
    syn = ants.registration(fixed=reference, moving=moving,
                            type_of_transform='SyN',
                            initial_transform=rigid['fwdtransforms'][0],
                            outprefix=str(output / 'syn_'))
    # Use exactly the list returned by SyN, as the research script does.
    # The Rigid initializer must not be appended a second time.
    transforms = syn['fwdtransforms']
    progress('Transforming Core with research genericLabel interpolation…')
    warped_core = ants.apply_transforms(fixed=reference, moving=read_rpi(core_path),
                                       transformlist=transforms, interpolator='genericLabel')
    # CBF is transformed for the unchanged alignment-review display only.
    warped_cbf = ants.apply_transforms(fixed=reference, moving=moving,
                                      transformlist=transforms, interpolator='linear')
    return reference, warped_cbf, warped_core, transforms


def measure_regions(mask, reference, atlases):
    """Count positive mask voxels, with the research reference-voxel volume."""
    mask_array = mask.numpy()
    indices = np.where(mask_array > 0)
    voxel_ml = float(np.prod(reference.spacing) / 1000.0)
    values = {'Total_Core_Vol': float(np.sum(mask_array > 0) * voxel_ml)}
    for name, atlas in atlases.items():
        ids, counts = np.unique(atlas.numpy()[indices], return_counts=True)
        values.update({f'{name}_{int(label)}_Core': float(count * voxel_ml)
                       for label, count in zip(ids, counts) if label > 0})
    return values
