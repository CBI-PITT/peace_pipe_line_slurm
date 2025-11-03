import os
import sys
import tifffile
import numpy as np

from pathlib import Path
this_script = Path(__file__)
operation_folder = this_script.parent
operations_folder = operation_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings
os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


def process_image(file, out_dir):
    def ideal_notch_filter(fshift, points):
        d0 = 121.0  # cutoff frequency
        H, W = fshift.shape
        u, v = np.ogrid[:H, :W]
        for d in range(len(points)):
            u0, v0 = points[d]
            mask1 = (u - u0) ** 2 + (v - v0) ** 2 <= d0
            mask2 = (u + u0) ** 2 + (v + v0) ** 2 <= d0
            fshift[mask1] = 0
            fshift[mask2] = 0
        return fshift

    def fft_2d_notch_filter(image, stripes_direction="v"):
        """
        stripes_direction: "h" (horizontal stripes) or "v" (vertical stripes)
        """
        image_dtype = image.dtype
        image = image.astype("float32")
        H, W = image.shape

        # do 2D fft
        img_fft = np.fft.fft2(image) / (W * H)
        # shift fft to get maximum at the center
        img_fft = np.fft.fftshift(img_fft)

        # filter shifted fft
        points = []
        image_center = (H // 2, W // 2)
        if stripes_direction == "h":
            # compute points on vertical axis of symmetry
            for point_ind in range(1, image_center[0] // first_harmonic):
                points.extend([
                    [image_center[0] + point_ind * first_harmonic, image_center[1]],
                    [image_center[0] - point_ind * first_harmonic, image_center[1]]
                ])
        elif stripes_direction == "v":
            # compute points on horizontal axis of symmetry
            for point_ind in range(1, image_center[1] // first_harmonic):
                points.extend([
                    [image_center[0], image_center[1] + point_ind * first_harmonic],
                    [image_center[0], image_center[1] - point_ind * first_harmonic]
                ])
        points = np.asarray(points)

        filtered_fft_shift = ideal_notch_filter(img_fft, points)

        # unshift
        img_fft = np.fft.ifftshift(filtered_fft_shift)
        # do inverse fft
        out_ifft = np.fft.ifft2(img_fft)
        # image = (np.real(out_ifft) * W * H).astype(image_dtype)
        image = (np.abs(out_ifft) * W * H).astype(image_dtype)  # works better for fMOST data
        return image

    img = tifffile.imread(file)
    img = fft_2d_notch_filter(img, stripes_direction=stripe_direction)
    tifffile.imwrite(os.path.join(out_dir, os.path.basename(file)), img)


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = sys.argv[3]
channel = sys.argv[4]
z = sys.argv[5]
stripe_direction = sys.argv[6]
first_harmonic = int(sys.argv[7])

input_file = os.path.join(
    INPUT_DIR,
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

process_image(input_file, OUTPUT_DIR)
