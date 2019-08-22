import numpy as np

# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def get_preprocessed_state(images, bit_depth):
    # Quantise to given bit depth and centre
    images = images // 2**(8 - bit_depth)
    images = images / 2**bit_depth
    images -= 0.5
    # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    images += np.random.rand(64, 64, 3) / 2**bit_depth
    return images

# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def get_postprocessed_state(images, bit_depth):
    images += 0.5
    images *= 2**bit_depth
    images = np.floor(images)
    images *= 2**(8 - bit_depth)
    return np.clip(images, 0, 2**8 - 1).astype(np.uint8)
