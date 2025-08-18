import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def kmeans_init(pixels, k):
    centroids = []
    centroids.append(pixels[np.random.randint(0, pixels.shape[0])])
    
    for _ in range(1, k):
        dist_sq = np.min(np.linalg.norm(pixels[:, None] - np.array(centroids)[None, :], axis=2)**2, axis=1)
        probs = dist_sq / dist_sq.sum()
        cumulative_probs = np.cumsum(probs)
        r = np.random.rand()
        next_centroid = pixels[np.searchsorted(cumulative_probs, r)]
        centroids.append(next_centroid)
    
    return np.array(centroids)


def kmeans_color_quantization(img, K=16, max_iters=20, visualize=False, log_file="L2_norm_log.txt"):
    # Normalize image to [0,1]
    img_norm = img.astype(np.float32) / 255.0
    data = img_norm.reshape((-1, 3))

    np.random.seed(42)
    centroids = kmeans_init(data, K)
    
    with open(log_file, "w") as f:
        for i in range(max_iters):
            distances = np.linalg.norm(data[:, None] - centroids[None, :], axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([
                data[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k] 
                for k in range(K)
            ])
            
            l2_norm_labels = np.linalg.norm(data - new_centroids[labels]).sum()
            f.write(f"Iteration {i+1}: L2 norm = {l2_norm_labels:.6f}\n")
            print(f"Iteration {i+1}: L2 norm = {l2_norm_labels:.6f}")
            
            if np.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            
            centroids = new_centroids

    quantized_data = centroids[labels]
    quantized_img_norm = quantized_data.reshape(img.shape)

    # Compute total L2 norm in [0,1] scale
    total_l2_norm = np.linalg.norm(img_norm - quantized_img_norm).sum()
    print(f"Evaluation: Total L2 norm between original and quantized image (0–1 scale) = {total_l2_norm:.6f}")
    
    # Convert back to uint8 for saving
    quantized_img = (quantized_img_norm * 255).astype(np.uint8)

    if visualize:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title(f"Quantized (K={K})")
        plt.imshow(cv2.cvtColor(quantized_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
    return quantized_img


if __name__ == "__main__":
    images = ["./data/lena.png", "./data/peppers.tif"]
    K = 16
    visualize = True  
    max_iters = 100

    for img_path in images:
        img = cv2.imread(img_path)
        quantized_img = kmeans_color_quantization(
            img, K=K, visualize=visualize, max_iters=max_iters,
            log_file=f"{os.path.splitext(os.path.basename(img_path))[0]}_L2_log.txt"
        )
        out_path = f"{os.path.splitext(img_path)[0]}_quantized.png"
        cv2.imwrite(out_path, quantized_img)
        print(f"Saved quantized image to {out_path}")
