import rawpy
import numpy as np
import scipy.ndimage
import imageio

def process_raw_image(path_in, path_out):
    # Step 1: Read the DNG file at half resolution
    with rawpy.imread(path_in, half_size=True) as raw:
        raw_image = raw.raw_image_visible.astype(np.float32)
    
    # Normalize to [0, 1] range
    raw_image /= 65535.0
    
    # Step 2: Create Bayer pattern kernels for demosaicing
    # Define kernels for red, green, and blue
    red_kernel = np.array([[1, 0], [0, 0]])
    green_kernel_1 = np.array([[0, 1], [1, 0]])
    green_kernel_2 = np.array([[1, 0], [0, 1]])
    blue_kernel = np.array([[0, 0], [0, 1]])

    # Apply kernels to the RAW image using scipy.ndimage.correlate
    red_channel = scipy.ndimage.correlate(raw_image, red_kernel, mode='reflect')
    green_channel_1 = scipy.ndimage.correlate(raw_image, green_kernel_1, mode='reflect')
    green_channel_2 = scipy.ndimage.correlate(raw_image, green_kernel_2, mode='reflect')
    blue_channel = scipy.ndimage.correlate(raw_image, blue_kernel, mode='reflect')
    
    # Step 3: Separate the channels using slicing and indexing (combine green channels)
    red = raw_image[::2, ::2]
    green = (green_channel_1 + green_channel_2) / 2
    blue = raw_image[1::2, 1::2]

    # Step 4: Stack red, green, and blue channels
    rgb_image = np.stack([red, green, blue], axis=-1)

    # Step 5: White balance (scale each channel so its mean is 0.25)
    for i in range(3):
        rgb_image[:, :, i] *= 0.25 / np.mean(rgb_image[:, :, i])

    # Step 6: Apply gamma correction
    gamma = 0.55
    rgb_image = np.power(rgb_image, gamma)

    # Step 7: Clip the values to [0, 1] and quantize to 8-bit integers
    rgb_image = np.clip(rgb_image, 0, 1)
    rgb_image = (rgb_image * 255).astype(np.uint8)

    # Step 8: Save the image as JPEG
    imageio.imwrite(path_out, rgb_image)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python hw1.py <input_dng_file> <output_jpeg_file>")
    else:
        path_in = sys.argv[1]
        path_out = sys.argv[2]
        process_raw_image(path_in, path_out)

