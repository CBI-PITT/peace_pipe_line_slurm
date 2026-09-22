# conda activate remove_bg_py311_cuda124
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from transformers import Sam2Model, Sam2Processor

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings
from utils.z_range import resolve_tiff_path

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


MODEL_ID = "facebook/sam2.1-hiera-small"


def robust_limits(image, p_low=1.0, p_high=99.8):
    """
    Estimate intensity normalization limits without copying an enormous array.
    """
    image = np.asarray(image)

    sy = max(1, image.shape[0] // 512)
    sx = max(1, image.shape[1] // 512)
    sample = image[::sy, ::sx]

    sample = sample[np.isfinite(sample)]

    lo, hi = np.percentile(sample, [p_low, p_high])

    if hi <= lo:
        hi = lo + 1.0

    return float(lo), float(hi)


def normalize_uint8(image, lo=None, hi=None):
    """
    Convert a grayscale microscopy image to uint8 using robust normalization.
    """
    image = np.asarray(image)

    if lo is None or hi is None:
        lo, hi = robust_limits(image)

    x = np.clip(image.astype(np.float32), lo, hi)
    x = (x - lo) / (hi - lo)
    x = (255.0 * x).astype(np.uint8)

    return x


def resize_keep_aspect(image, max_side=1024, resample=Image.Resampling.BILINEAR):
    """
    Resize while preserving aspect ratio.

    Returns
    -------
    resized : np.ndarray
    scale : float
        resized_size / original_size
    """
    h, w = image.shape[:2]

    scale = min(1.0, max_side / max(h, w))

    if scale == 1.0:
        return np.asarray(image), scale

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    pil = Image.fromarray(image)
    pil = pil.resize((new_w, new_h), resample=resample)

    return np.asarray(pil), scale


def largest_component(mask):
    """
    Keep only the largest connected component of a 2D binary mask.
    """
    labels, n = ndi.label(mask)

    if n == 0:
        return mask.astype(bool)

    sizes = ndi.sum(mask, labels, range(1, n + 1))
    largest = np.argmax(sizes) + 1

    return labels == largest


def rough_brain_box(
    image_u8,
    pad_fraction=0.05,
    gaussian_sigma=3.0,
    bright_foreground=True,
):
    """
    Obtain an approximate brain bounding box.

    This is only used as a prompt for SAM2; it is NOT the final mask.

    Returns
    -------
    box : [x1, y1, x2, y2]
    """
    image_u8 = np.asarray(image_u8)

    smooth = ndi.gaussian_filter(
        image_u8.astype(np.float32),
        sigma=gaussian_sigma,
    )

    try:
        thresh = threshold_otsu(smooth)
    except ValueError:
        h, w = image_u8.shape
        return [0, 0, w - 1, h - 1]

    if bright_foreground:
        mask = smooth > thresh
    else:
        mask = smooth < thresh

    # Remove isolated pixels.
    mask = ndi.binary_opening(mask, iterations=2)
    mask = ndi.binary_closing(mask, iterations=3)

    mask = largest_component(mask)
    mask = ndi.binary_fill_holes(mask)

    yy, xx = np.where(mask)

    h, w = image_u8.shape

    if len(xx) == 0:
        return [0, 0, w - 1, h - 1]

    x1 = int(xx.min())
    x2 = int(xx.max())
    y1 = int(yy.min())
    y2 = int(yy.max())

    pad_x = int((x2 - x1) * pad_fraction)
    pad_y = int((y2 - y1) * pad_fraction)

    x1 = max(0, x1 - pad_x)
    x2 = min(w - 1, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(h - 1, y2 + pad_y)

    return [x1, y1, x2, y2]


def clean_2d_mask(mask):
    """
    Simple cleanup appropriate for one large brain object.
    """
    mask = mask.astype(bool)
    mask = largest_component(mask)
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_closing(mask, iterations=2)

    return mask


def upscale_mask_smooth(
    mask,
    output_shape,
    edge_smoothing_px=0,
    return_soft=False,
):
    """
    Upscale a low-resolution binary mask to full resolution.

    edge_smoothing_px is a Gaussian smoothing sigma in FULL-RESOLUTION pixels:
    0 = no smoothing, 2-5 = subtle, 5-15 = useful for large microscopy images,
    15-30 = strong smoothing.
    """
    h, w = output_shape

    # Bilinear interpolation already removes much of the pixelated appearance.
    pil = Image.fromarray(mask.astype(np.uint8) * 255)

    pil = pil.resize(
        (w, h),
        resample=Image.Resampling.BILINEAR,
    )

    soft_mask = np.asarray(pil, dtype=np.float32) / 255.0

    if edge_smoothing_px > 0:
        soft_mask = ndi.gaussian_filter(
            soft_mask,
            sigma=edge_smoothing_px,
        )

    soft_mask = np.clip(soft_mask, 0.0, 1.0)

    if return_soft:
        return soft_mask

    return soft_mask >= 0.5


def load_sam2_2d(model_id=MODEL_ID):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = Sam2Model.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    processor = Sam2Processor.from_pretrained(model_id)

    return model, processor, device


def segment_brain_2d(
    image,
    model,
    processor,
    device,
    working_max_side=1536,
    box=None,
    bright_foreground=True,
    edge_smoothing_px=2,
    feather_edge=True,
):
    """
    Segment the brain in a large 2D microscopy image.

    Returns
    -------
    mask : bool ndarray
        Final binary brain mask.

    masked_image : ndarray
        Background-removed image with same dtype as input.
    """

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, got shape {image.shape}"
        )

    original_dtype = image.dtype

    # Normalize only for SAM.
    lo, hi = robust_limits(image)

    image_u8 = normalize_uint8(
        image,
        lo,
        hi,
    )

    # Downsample for SAM.
    small, scale = resize_keep_aspect(
        image_u8,
        max_side=working_max_side,
    )

    small_rgb = np.repeat(
        small[..., None],
        3,
        axis=-1,
    )

    # Bounding box prompt for SAM2.
    if box is None:
        box_small = rough_brain_box(
            small,
            bright_foreground=bright_foreground,
        )
    else:
        box_small = [
            int(round(box[0] * scale)),
            int(round(box[1] * scale)),
            int(round(box[2] * scale)),
            int(round(box[3] * scale)),
        ]

    print("SAM2 working image:", small.shape)
    print("SAM2 box:", box_small)

    input_boxes = [[box_small]]

    inputs = processor(
        images=small_rgb,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(device)

    # SAM2 inference.
    with torch.inference_mode():

        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(
                    **inputs,
                    multimask_output=False,
                )

        else:
            outputs = model(
                **inputs,
                multimask_output=False,
            )

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
    )[0]

    mask_small = masks[0, 0].numpy() > 0

    mask_small = clean_2d_mask(mask_small)

    # Upscale and smooth the boundary.
    soft_mask = upscale_mask_smooth(
        mask_small,
        output_shape=image.shape,
        edge_smoothing_px=edge_smoothing_px,
        return_soft=True,
    )

    # Binary version, useful for saving / subsequent processing.
    mask = soft_mask >= 0.5

    mask = largest_component(mask)
    mask = ndi.binary_fill_holes(mask)

    # Apply to ORIGINAL image.
    if feather_edge and edge_smoothing_px > 0:
        # The soft mask produces a smooth transition only at the boundary.
        result = (
            image.astype(np.float32)
            * soft_mask
        )

        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            result = np.clip(
                np.rint(result),
                info.min,
                info.max,
            )

        masked_image = result.astype(
            original_dtype
        )

    else:
        # Hard background cutoff.
        masked_image = image.copy()
        masked_image[~mask] = 0

    return mask, masked_image


INPUT_DIR = sys.argv[1]
IMG_OUT_DIR = sys.argv[2]
MASK_OUT_DIR = sys.argv[3]
resolution_level = sys.argv[4]
channel = sys.argv[5]
z = sys.argv[6]

print("INPUT_DIR", INPUT_DIR)
print("IMG_OUT_DIR", IMG_OUT_DIR)
print("MASK_OUT_DIR", MASK_OUT_DIR)
print("resolution_level", resolution_level)
print("channel", channel)
print("z", z)

input_file = resolve_tiff_path(INPUT_DIR, None, resolution_level, channel, z)

output_img_file = os.path.join(
    IMG_OUT_DIR,
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

output_mask_file = os.path.join(
    MASK_OUT_DIR,
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

print("Removing background")
image = tifffile.imread(input_file)

model, processor, device = load_sam2_2d()

mask, brain_only = segment_brain_2d(
    image,
    model,
    processor,
    device,

    # 1024 is usually enough.
    # 1536 gives the preprocessing a bit more spatial information.
    working_max_side=1536,
)

tifffile.imwrite(output_mask_file, mask.astype(np.uint8) * 255)
tifffile.imwrite(output_img_file, brain_only)
print("Done")
