import cv2
import numpy as np
from math import log10
import matplotlib.pyplot as plt

watermarked_path = r"C:\Users\mahdi\OneDrive\Desktop\TechStack\Tasks\techstack2025-ai\week1\Task1\data\watermarked.png"
watermark_path = r"C:\Users\mahdi\OneDrive\Desktop\TechStack\Tasks\techstack2025-ai\week1\Task1\data\iut.png"
original_path = r"C:\Users\mahdi\OneDrive\Desktop\TechStack\Tasks\techstack2025-ai\week1\Task1\data\original.png"

def dewatermark(watermarked, watermark):
    watermark_rgb = watermark[:,:, :3] 
    alpha = watermark[:,:, 3]
    dn = 1 - alpha * 0.3
    dn = np.where(dn == 0, 1e-9, dn)
    dewatermarked = (watermarked - alpha[..., np.newaxis] * 0.3 * watermark_rgb) / dn[..., np.newaxis]
    dewatermarked = np.clip(dewatermarked, 0, 1)
    return dewatermarked

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def calculate_psnr(original, compared):
    mse_val = mse(original, compared)
    if mse_val == 0:
        return float('inf')
    return 20 * np.log10(255 / np.sqrt(mse_val))

watermarked = plt.imread(watermarked_path).astype(np.float32)
watermark = plt.imread(watermark_path).astype(np.float32)
original = plt.imread(original_path).astype(np.float32)

dewatermarked = dewatermark(watermarked, watermark)
psnr_value = calculate_psnr(original, dewatermarked )
plt.imshow(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB))
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(dewatermarked, cv2.COLOR_BGR2RGB))
plt.title('Dewatermarked Image')
plt.axis('off')
plt.show()

print(f"PSNR between dewatermarked and original: {psnr_value:.2f} dB")