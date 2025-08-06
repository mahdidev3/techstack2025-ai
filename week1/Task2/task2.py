import os
import glob
import time 

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

folder_path = r"C:\Users\mahdi\OneDrive\Desktop\TechStack\Tasks\techstack2025-ai\week1\Task2\data\Puzzle_1_160"
output_file = os.path.join(folder_path, "Output.tif")
patch_files = glob.glob(os.path.join(folder_path, "Patch_*.tif"))
corners_img = np.array(Image.open(output_file))
base_img = corners_img.copy()
patches = [np.array(Image.open(patch_file)) for patch_file in patch_files]


print(base_img.shape , len(patches) , patches[0].shape)

rows, cols = base_img.shape[0] // patches[0].shape[0] , base_img.shape[1] // patches[0].shape[0]
img_width, img_height = 1920, 1200
patch_width, patch_height = img_width // cols, img_height // rows

def puzzle_solver(base_img , patches):
    final_img = base_img.copy()
    solved = np.zeros((rows, cols), dtype=bool)
    solved[0][0] = solved[rows - 1][0] = solved[0][cols - 1] = solved[rows - 1][cols - 1] = True
    def get_edge_diff(patch1, patch2, direction):
        if direction == 'top':
            edge1 = patch1[:1, :]
            edge2 = patch2[-1:, :]
        elif direction == 'bottom':
            edge1 = patch1[-1:, :]
            edge2 = patch2[:1, :]
        elif direction == 'left':
            edge1 = patch1[:, :1]
            edge2 = patch2[:, -1:]
        elif direction == 'right':
            edge1 = patch1[:, -1:]
            edge2 = patch2[:, :1]
        return np.sum((edge1 - edge2) ** 2)

    used_patches = set()

    plt.ion()
    fig, ax = plt.subplots()

    while len(used_patches) < len(patches):
        min_diff = float('inf')
        best_patch = None
        best_idx = -1
        best_row, best_col = -1, -1
        
        for row in range(rows):
            for col in range(cols):
                if solved[row][col]:
                    continue

                neighbors = [
                    ('left', (row, col-1)) if col > 0 and solved[row][col-1] else None,
                    ('right', (row, col+1)) if col < cols-1 and solved[row][col+1] else None,
                    ('top', (row-1, col)) if row > 0 and solved[row-1][col] else None,
                    ('bottom', (row+1, col)) if row < rows-1 and solved[row+1][col] else None
                ]
                neighbors = [n for n in neighbors if n]
                if not neighbors:
                    continue
                
                for idx, patch in enumerate(patches):
                    if idx in used_patches:
                        continue
                    total_diff = 0
                    count = 0
                    
                    for direction, (n_row, n_col) in neighbors:
                        neighbor_patch = final_img[n_row*patch_height:(n_row+1)*patch_height, 
                                                n_col*patch_width:(n_col+1)*patch_width]
                        total_diff += get_edge_diff(patch, neighbor_patch, direction)
                        count += 1
                    
                    avg_diff = total_diff / count if count > 0 else float('inf')
                    if avg_diff < min_diff:
                        min_diff = avg_diff
                        best_patch = patch
                        best_idx = idx
                        best_row, best_col = row, col
        
        if best_patch is not None:
            final_img[best_row*patch_height:(best_row+1)*patch_height, 
                    best_col*patch_width:(best_col+1)*patch_width] = best_patch
            solved[best_row][best_col] = True
            used_patches.add(best_idx)
            
            ax.clear()
            ax.imshow(final_img)
            ax.axis('off')
            plt.draw()
            plt.pause(0.1)
    
    return final_img


final_img = puzzle_solver(base_img, patches)
Image.fromarray(final_img).save(os.path.join(folder_path, "final.tif"))


def accuracy_block_base(original_img, final_img, rows, cols):
    correct_blocks = 0
    total_blocks = rows * cols 
    for row in range(rows):
        for col in range(cols):
            patch = final_img[row*patch_height:(row+1)*patch_height, col*patch_width:(col+1)*patch_width]
            original_patch = original_img[row*patch_height:(row+1)*patch_height, col*patch_width:(col+1)*patch_width]

            if np.array_equal(patch, original_patch):
                correct_blocks += 1
    
    return 100 * correct_blocks / total_blocks

original_img = np.array(Image.open(os.path.join(folder_path, "Original.tif")))
accuracy= accuracy_block_base(original_img, final_img, rows, cols)
print(f"Accuracy: {accuracy:.2f}%")