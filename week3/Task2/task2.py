import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

def setup_logging(image_path):
    result_dir = './results'
    os.makedirs(result_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = os.path.basename(image_path).replace('.png', '')
    log_file = os.path.join(result_dir, f'seam_carving_{image_name}_{timestamp}.log')
    
    # Create a unique logger for this image
    logger = logging.getLogger(image_name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicate logging
    logger.handlers = []
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Create stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

def normalize(img):
    img = img.astype(np.float64)
    if img.max() > 0:
        img = img / img.max()
    return img

def compute_energy_map(image, saliency=None, depth=None,
                      w_sobel=0.33, w_saliency=0.33, w_depth=0.34):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) 
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = normalize(sobel_mag)

    energy = w_sobel * sobel_mag

    if saliency is not None:
        saliency = normalize(saliency)
        energy += w_saliency * saliency

    if depth is not None:
        depth = normalize(depth)
        energy += w_depth * depth

    return energy

def find_vertical_seam(energy_map):
    rows, cols = energy_map.shape
    dp_prev = energy_map[0].copy()
    parent = np.zeros((rows, cols), dtype=np.int32)

    for i in range(1, rows):
        padded = np.pad(dp_prev, (1, 1), mode="constant", constant_values=np.inf)
        left = padded[:-2]
        up = padded[1:-1]
        right = padded[2:]
        min_choice = np.minimum(np.minimum(left, up), right)
        choice = np.argmin(np.stack([left, up, right]), axis=0) - 1
        dp_curr = energy_map[i] + min_choice
        parent[i] = np.arange(cols) + choice
        dp_prev = dp_curr

    seam = np.zeros(rows, dtype=np.int32)
    seam[-1] = np.argmin(dp_prev)
    for i in range(rows-2, -1, -1):
        seam[i] = parent[i+1, seam[i+1]]

    return [(i, seam[i]) for i in range(rows)]

def remove_seam(image, saliency, depth, seam):
    rows, cols, channels = image.shape
    new_image = np.zeros((rows, cols-1, channels), dtype=image.dtype)
    new_saliency = np.zeros((rows, cols-1), dtype=image.dtype)
    new_depth = np.zeros((rows, cols-1), dtype=image.dtype)

    for i, j in seam:
        new_image[i, :j, :] = image[i, :j, :]
        new_image[i, j:, :] = image[i, j+1:, :]
        new_saliency[i, :j] = saliency[i, :j]
        new_saliency[i, j:] = saliency[i, j+1:]
        new_depth[i, :j] = depth[i, :j]
        new_depth[i, j:] = depth[i, j+1:]

    return new_image, new_saliency, new_depth

def seam_carving(image_path, seams_to_remove, visualize=False, saliency_map_path=None, depth_map_path=None):
    logger = setup_logging(image_path)
    
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Cannot load image at {image_path}")
        raise ValueError(f"Cannot load image at {image_path}")
    
    saliency = None
    if saliency_map_path:
        saliency = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)
        if saliency is None:
            logger.error(f"Cannot load saliency map at {saliency_map_path}")
            raise ValueError(f"Cannot load saliency map at {saliency_map_path}")
    
    depth = None
    if depth_map_path:
        depth = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
        if depth is None:
            logger.error(f"Cannot load depth map at {depth_map_path}")
            raise ValueError(f"Cannot load depth map at {depth_map_path}")

    energy_map = compute_energy_map(image, saliency, depth)
    
    plt.ion()
    fig, ax = plt.subplots()
    
    for step in range(seams_to_remove):
        seam = find_vertical_seam(energy_map)
        seam_energy = sum(energy_map[i, j] for i, j in seam)
        logger.info(f"Step {step + 1}: Removed seam with total energy {seam_energy:.4f}")
        logger.info(f"Energy map mean: {energy_map.mean():.4f}, max: {energy_map.max():.4f}")
        
        if visualize:
            img_copy = image.copy()
            for i, j in seam:
                img_copy[i, j, :] = [0, 0, 255]
            ax.clear()
            ax.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Seam {step + 1}/{seams_to_remove}")
            plt.draw()
            plt.pause(0.001)
        
        image, saliency, depth = remove_seam(image, saliency, depth, seam)
        energy_map = compute_energy_map(image, saliency, depth)
    
    plt.ioff()
    plt.close()
    
    output_path = os.path.join('results', os.path.basename(image_path).replace('.png', '_resized.png'))
    cv2.imwrite(output_path, image)
    logger.info(f"Saved output image to {output_path}")
    
    return image

def main():
    images = ['./data/Snowman/Snowman.png', './data/Baby/Baby.png', './data/Diana/Diana.png']
    seams_to_remove = 200
    visualize = True
    
    for img in images:
        logger = setup_logging(img)
        logger.info(f"Processing {img}...")
        seam_carving(img, seams_to_remove, visualize=visualize,
                    saliency_map_path=img.replace('.png', '_SMap.png'),
                    depth_map_path=img.replace('.png', '_DMap.png'))

if __name__ == '__main__':
    main()