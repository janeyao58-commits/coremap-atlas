"""Research registration behind the unchanged CoreMap interface."""
from research_processing import register_core
from pathlib import Path
import json
import numpy as np
import nibabel as nib
from predictor import find, quantify


def register(cbf_path, core_path, out, side, progress=lambda message: None):
    import ants
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cbf, core = nib.load(str(cbf_path)), nib.load(str(core_path))
    for image in (cbf, core):
        if len(image.shape) != 3 or image.header.get_xyzt_units()[0] != 'mm':
            raise ValueError('CBF and Core must be 3D NIfTI images with mm units.')
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
            raise ValueError('Invalid image geometry.')
    if cbf.shape != core.shape or not np.allclose(cbf.affine, core.affine, atol=1e-4):
        raise ValueError('Native CBF and Core must share the same acquisition grid.')
    data = cbf.get_fdata(dtype=np.float32)
    mask = core.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError('CBF must be finite and Core must be a finite binary 0/1 mask.')
    native_ml = float(np.sum(mask > 0) * abs(np.linalg.det(core.affine[:3, :3])) / 1000.0)
    has_core = bool(mask.any())
    del data, mask
    reference, warped_cbf, warped_core, transforms = register_core(
        cbf_path, core_path, find('MNI152_T1_1mm_brain.nii.gz'), out, progress)
    cbf_out, core_out = out/'CBF_MNI.nii.gz', out/'Core_MNI.nii.gz'
    ants.image_write(warped_cbf, str(cbf_out))
    ants.image_write(warped_core, str(core_out))
    del reference, warped_cbf, warped_core
    for path in (cbf_out, core_out):
        image = nib.load(str(path))
        image.header.set_xyzt_units('mm')
        nib.save(image, str(path))
    progress('Building alignment review images…')
    qc_images(cbf_out, core_out, out)
    result = quantify(core_out, side)
    if has_core and result['total'] == 0:
        raise ValueError('Core disappeared during transformation. Registration requires review; prediction is blocked.')
    result.update(native_core_ml=native_ml, requires_visual_review=True,
                  registration='Research RPI; Rigid then SyN; MNI152 1-mm reference; genericLabel core interpolation',
                  transforms=transforms)
    (out/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def qc_images(cbf_path, core_path, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref = nib.as_closest_canonical(nib.load(str(find('MNI152_T1_1mm_brain.nii.gz'))))
    t, c, m = ref.get_fdata(), nib.as_closest_canonical(nib.load(str(cbf_path))).get_fdata(), nib.as_closest_canonical(nib.load(str(core_path))).get_fdata()
    top = np.percentile(c[c>0],98) if (c>0).any() else 1
    # Every 8 mm throughout the template, including superior and inferior slices.
    zs = list(range(-48,81,8))
    fig, axes = plt.subplots(3,6,figsize=(15,8), facecolor='black')
    for ax,z in zip(axes.flat,zs):
        k = int(round(nib.affines.apply_affine(np.linalg.inv(ref.affine), [0,0,z])[2]))
        ax.imshow(t[:,:,k].T, origin='lower', cmap='gray')
        ax.imshow(np.ma.masked_less_equal(c[:,:,k].T,0), origin='lower', cmap='viridis', vmin=0,vmax=top,alpha=.45)
        if m[:,:,k].any(): ax.contour(m[:,:,k].T,levels=[.5],colors=['red'],linewidths=.7)
        ax.set_title(f'z = {z} mm',color='white',fontsize=9); ax.axis('off')
    for ax in list(axes.flat)[len(zs):]: ax.axis('off')
    fig.tight_layout(); fig.savefig(out/'qc.png',dpi=130,facecolor='black'); plt.close(fig)

