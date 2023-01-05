def fft_cluster(options):
    import os
    from glob import glob
    import dask
    from dask.distributed import Client
    client = Client('10.240.0.99:8786')
    # client = Client('10.240.0.99:9090')

    def process_image(file, first_harmonic, out_dir):
        import os
        import tifffile
        import numpy as np

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
        img = fft_2d_notch_filter(img)
        tifffile.imwrite(os.path.join(out_dir, os.path.basename(file)), img)
        return True

    analysis_path = os.path.join(options["out_name"], 'resolution_level_x')

    for channel in range(options["channels"]):
        in_dir = os.path.join(analysis_path, f"channel_{channel}")
        fft_out_dir = os.path.join(analysis_path, f"channel_{channel}_notch_filtered")
        if not os.path.exists(fft_out_dir):
            os.makedirs(fft_out_dir)
        imgs = sorted(glob(os.path.join(in_dir, "*.tif")))
        imgs = [i for i in imgs if not os.path.exists(os.path.join(fft_out_dir, os.path.basename(i)))]
        processed = [dask.delayed(process_image)(i, options["first_harmonic"], fft_out_dir) for i in imgs]
        processed = dask.compute(processed, threads_per_worker=2)
