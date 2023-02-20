import numpy as np


def denoise_fft(image):
    """
    Apply circular mask to the image in FFT domain.
    """
    img_float = image.astype(np.float32)

    H, W = img_float.shape
    img_fft = np.fft.fft2(img_float)/(W * H)

    img_fft = np.fft.fftshift(img_fft)

    center = [H//2, W//2]
    r = 500  # TODO hardcoded 500
    x, y = np.ogrid[:H, :W]
    mask_area = (x - center[0])**2 + (y - center[1])**2 >= r**2
    img_fft[mask_area] = 0

    img_fft = np.fft.ifftshift(img_fft)
    out_ifft = np.fft.ifft2(img_fft)

    image = (np.real(out_ifft) * W * H).astype(np.uint16)
    return image


def denoise_fft_ellipse(image):
    """
    Apply ellipse-like mask to the image in FFT domain, instead of circle.

    Coronal plane images have small height (few z layers) and large width (many x points).
    It makes sense to use different cutoff frequencies for z and x directions.
    """
    import cv2

    img_float = image.astype(np.float32)

    H, W = img_float.shape
    img_fft = np.fft.fft2(img_float) / (W * H)

    img_fft = np.fft.fftshift(img_fft)

    mask = np.zeros_like(image).astype(np.uint8)
    new_mask = cv2.ellipse(mask, (W // 2, H // 2), (200, 50), 0, 0, 360, 1, -1)  # TODO hardcoded (200, 50)
    mask_area = np.where(new_mask == 0)

    img_fft[mask_area] = 0

    img_fft = np.fft.ifftshift(img_fft)
    out_ifft = np.fft.ifft2(img_fft)

    image = (np.real(out_ifft) * W * H).astype(np.uint16)
    return image
