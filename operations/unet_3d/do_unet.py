import os
import sys

import torch
import torch.nn as nn
import tifffile
import numpy as np
from torchvision.transforms import functional as F
import torch.nn.functional as G
from torchvision.transforms import ToTensor
# from skimage import measure
# import pandas as pd

# from analysis import settings


input_file = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = sys.argv[3]
channel = sys.argv[4]
model = sys.argv[5]

print("OUTPUT_DIR", OUTPUT_DIR)
print("resolution_level", resolution_level)
print("channel", channel)
print("model", model)


def segment(stack, model_path):
    class UNet3D(nn.Module):
        def __init__(self, in_channels=1, out_channels=1):
            super(UNet3D, self).__init__()

            # Contracting path
            self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm3d(64)
            self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm3d(128)
            self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm3d(256)
            self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm3d(512)
            self.conv5 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm3d(1024)

            # Expanding path
            self.upconv6 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
            self.conv6 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
            self.bn6 = nn.BatchNorm3d(512)
            self.upconv7 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
            self.conv7 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
            self.bn7 = nn.BatchNorm3d(256)
            self.upconv8 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
            self.conv8 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
            self.bn8 = nn.BatchNorm3d(128)
            self.upconv9 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
            self.conv9 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
            self.bn9 = nn.BatchNorm3d(64)

            # Output layer
            self.output = nn.Conv3d(64, out_channels, kernel_size=1)

        def forward(self, x):
            # Contracting path
            conv1 = G.relu(self.bn1(self.conv1(x)))
            conv2 = G.relu(self.bn2(self.conv2(G.max_pool3d(conv1, kernel_size=2, stride=2))))
            conv3 = G.relu(self.bn3(self.conv3(G.max_pool3d(conv2, kernel_size=2, stride=2))))
            conv4 = G.relu(self.bn4(self.conv4(G.max_pool3d(conv3, kernel_size=2, stride=2))))
            conv5 = G.relu(self.bn5(self.conv5(G.max_pool3d(conv4, kernel_size=2, stride=2))))

            # Expanding path
            upconv6 = self.upconv6(conv5)
            conv6 = G.relu(self.bn6(self.conv6(torch.cat([upconv6, conv4], dim=1))))
            upconv7 = self.upconv7(conv6)
            conv7 = G.relu(self.bn7(self.conv7(torch.cat([upconv7, conv3], dim=1))))
            upconv8 = self.upconv8(conv7)
            conv8 = G.relu(self.bn8(self.conv8(torch.cat([upconv8, conv2], dim=1))))
            upconv9 = self.upconv9(conv8)
            conv9 = G.relu(self.bn9(self.conv9(torch.cat([upconv9, conv1], dim=1))))

            # Output layer
            output = self.output(conv9)

            return output

    def normalize_image_stack(image_stack):
        mean = np.mean(image_stack)
        std = np.std(image_stack)
        normalized_stack = (image_stack - mean) / std
        return normalized_stack


    print("-------------------- SEGMENTATION ------------------")
    # chunk_number = sys.argv[2]
    # detection_folder = os.path.join(OUTPUT_DIR, f'scale_{SCALE}', 'detection')
    # try:
    #     os.makedirs(detection_folder)
    # except FileExistsError:
    #     pass

    # masks_folder = os.path.join(OUTPUT_DIR, f'scale_{SCALE}', 'detection_masks')
    # try:
    #     os.makedirs(masks_folder)
    # except FileExistsError:
    #     pass

    # out_filename = os.path.join(masks_folder, f'mask_chunk_{str(chunk_number).zfill(5)}.tif')
    # centroids_filename = os.path.join(detection_folder, f"napari_chunk_{str(chunk_number).zfill(5)}.csv")

    print("Chunk shape", stack.shape)
    stack = stack.astype('float32')

    # preprocess chunk
    # stack = normalize_image_stack(stack)

    volume_size = stack.shape
    patch_size = (128, 128, 128)
    step_size = (128, 128, 128)

    # Calculate the padding required in each dimension
    padding = [
        ((step_size[dim] - ((volume_size[dim] - patch_size[dim]) % step_size[dim])) % step_size[dim])
        for dim in range(3)
    ]

    # Apply padding to the input volume separately for each dimension
    padded_stack = np.pad(stack, ((0, padding[0]), (0, padding[1]), (0, padding[2])), mode='linear_ramp')

    def prediction(model, padded_stack, patch_size, threshold):
        # Initialize segmented stack shape
        segm_stack = np.zeros(padded_stack.shape[:3])
        segm_stack_th1 = np.zeros(padded_stack.shape[:3])

        # Predict each 3D patch
        patch_num = 1
        for i in range(0, padded_stack.shape[0], patch_size[0]):
            for j in range(0, padded_stack.shape[1], patch_size[1]):
                for k in range(0, padded_stack.shape[2], patch_size[2]):
                    single_patch = padded_stack[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
                    single_patch = normalize_image_stack(single_patch)
                    # Apply ToTensor() transform
                    # transform = ToTensor()
                    # stack_tensor = transform(single_patch)
                    stack_tensor = torch.from_numpy(single_patch)
                    stack_tensor = stack_tensor.unsqueeze(0)
                    stack_tensor = stack_tensor.unsqueeze(0)

                    # Pass the tensor through the model to obtain predictions
                    with torch.no_grad():
                        stack_tensor = stack_tensor.to(device, dtype=torch.float32)
                        prediction = model.forward(stack_tensor)
                        single_patch_prediction = (
                                    torch.sigmoid(prediction) > threshold).float()  # binarize with threshold of 0.5
                        # single_patch_prediction = (prediction > 0.5)
                        single_patch_prediction = single_patch_prediction.squeeze().cpu().numpy()
                        single_patch_prediction = np.interp(single_patch_prediction, (
                        single_patch_prediction.min(), single_patch_prediction.max()), (0, 255))
                        single_patch_prediction = np.array(single_patch_prediction)
                        # print(single_patch_prediction.shape)

                        single_patch_prediction_th1 = (
                                    torch.sigmoid(prediction) == 1).float()  # binarize with threshold of 0.5
                        # single_patch_prediction = (prediction > 0.9)
                        single_patch_prediction_th1 = single_patch_prediction_th1.squeeze().cpu().numpy()
                        single_patch_prediction_th1 = np.interp(single_patch_prediction_th1, (
                        single_patch_prediction_th1.min(), single_patch_prediction_th1.max()), (0, 255))
                        single_patch_prediction_th1 = np.array(single_patch_prediction_th1)

                    # Insert segmented small patch into the large patch at corresponding coordinates
                    segm_stack[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += single_patch_prediction
                    segm_stack_th1[i:i + patch_size[0], j:j + patch_size[1],
                    k:k + patch_size[2]] += single_patch_prediction_th1
                    print("Finished processing patch number ", patch_num, " at position ", i, j, k)
                    patch_num += 1

        return segm_stack, segm_stack_th1

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model weights and create the model architecture
    model = UNet3D().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # segmented_stack = prediction(model, padded_stack, patch_size, 0.999)
    # segmented_stack_th1 = prediction(model, padded_stack, patch_size, 1)

    segmented_stack, segmented_stack_th1 = prediction(model, padded_stack, patch_size, 0.999)

    # Remove the padding from the predicted output
    segmented_stack = segmented_stack[0:volume_size[0], 0:volume_size[1], 0:volume_size[2]]
    segmented_stack_th1 = segmented_stack_th1[0:volume_size[0], 0:volume_size[1], 0:volume_size[2]]

    # labels = measure.label(segmented_stack)
    # labels_th1 = measure.label(segmented_stack_th1)
    #
    # # Calculate the centroid coordinates of each connected component
    # table = pd.DataFrame(
    #     measure.regionprops_table(
    #         labels,
    #         properties=['label',
    #                     'coords']
    #     )
    # )
    #
    # # Calculate the centroid coordinates of each connected component
    # table_th1 = pd.DataFrame(
    #     measure.regionprops_table(
    #         labels_th1,
    #         properties=['label',
    #                     'coords']
    #     )
    # )
    #
    # # Precompute sets of coordinates
    # coords_dict = {label: set(tuple(coords) for coords in coords_list) for label, coords_list in
    #                zip(table['label'], table['coords'])}
    # coords_th1_dict = {label: set(tuple(coords) for coords in coords_list) for label, coords_list in
    #                    zip(table_th1['label'], table_th1['coords'])}
    #
    # # Iterate through labels and optimize comparisons
    # for label in table['label']:
    #     nuclei_coords = coords_dict[label]
    #     counter = 0
    #
    #     for label_th1 in table_th1['label']:
    #         nuclei_coords_th1 = coords_th1_dict[label_th1]
    #
    #         # Check if 'nuclei_coords' is a subset of 'nuclei_coords_th1'
    #         if nuclei_coords_th1.issubset(nuclei_coords):
    #             counter += 1
    #
    #     if counter >= 3:
    #         # Update 'segmented_stack' based on the label
    #         x, y, z = zip(*nuclei_coords)  # Extract coordinates
    #         segmented_stack[x, y, z] = 0

    segmented_stack = segmented_stack + segmented_stack_th1
    reconstructed_image = segmented_stack.astype(np.uint8)
    print(reconstructed_image.dtype)

    return reconstructed_image


output_file = os.path.join(
    OUTPUT_DIR,
    f"mask_{os.path.basename(input_file)}"
)

if os.path.exists(output_file):
    print(f"file {output_file} already exists")
    sys.exit(0)

print("Running 3D UNet")
img = tifffile.imread(input_file)
img = segment(img, model)
tifffile.imwrite(output_file, img)
print("Done")
