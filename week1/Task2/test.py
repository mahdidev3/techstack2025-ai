import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

base_data_path = r"C:\Users\mahdi\OneDrive\Desktop\TechStack\Tasks\techstack2025-ai\week1\Task2\data"

start_puzzle = 1  # change as needed
end_puzzle = 20    # change as needed
slot_types = [40,160]  # the two slot types

def puzzle_solver(base_img , patches):
    final_img = base_img.copy()
    solved = np.zeros((rows, cols), dtype=bool)
    solved[0][0] = solved[rows - 1][0] = solved[0][cols - 1] = solved[rows - 1][cols - 1] = True
    def get_edge_diff(patch1, patch2, direction , depth=1 , decay=0.0):

        total_diff = 0.0
        weghted_diff = 1

        for offset in range(depth):
            if direction == 'top':
                edge1 = patch1[:1 + offset, :]
                edge2 = patch2[-(1 + offset):, :]
            elif direction == 'bottom':
                edge1 = patch1[-(1 + offset):, :]
                edge2 = patch2[:1 + offset, :]
            elif direction == 'left':
                edge1 = patch1[:, :1 + offset]
                edge2 = patch2[:, -(1 + offset):]
            elif direction == 'right':
                edge1 = patch1[:, -(1 + offset):]
                edge2 = patch2[:, :1 + offset]
            total_diff += (np.sum((edge1 - edge2) ** 2) * weghted_diff)
            weghted_diff *= decay

        return total_diff / depth if depth > 0 else float('inf')

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
                
                min_diff_for_neighbors = float('inf')
                best_patch_for_neighbors = None
                best_idx_for_neighbors = -1
                best_row_for_neighbors, best_col_for_neighbors = -1, -1

                pre_min_diff_for_neighbors = float('inf')
                pre_best_patch_for_neighbors = None
                pre_best_idx_for_neighbors = -1
                pre_best_row_for_neighbors, pre_best_col_for_neighbors = -1, -1

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
                    if avg_diff < min_diff_for_neighbors:
                        pre_min_diff_for_neighbors = min_diff_for_neighbors
                        pre_best_patch_for_neighbors = best_patch_for_neighbors
                        pre_best_idx_for_neighbors = best_idx_for_neighbors
                        pre_best_row_for_neighbors, pre_best_col_for_neighbors = best_row_for_neighbors, best_col_for_neighbors

                        min_diff_for_neighbors = avg_diff
                        best_patch_for_neighbors = patch
                        best_idx_for_neighbors = idx
                        best_row_for_neighbors, best_col_for_neighbors = row, col


                if (pre_best_patch_for_neighbors is not None) and (pre_min_diff_for_neighbors <= (min_diff_for_neighbors * 1.01)):
                    continue

                if( min_diff_for_neighbors < min_diff):
                    min_diff = min_diff_for_neighbors
                    best_patch = best_patch_for_neighbors
                    best_idx = best_idx_for_neighbors
                    best_row, best_col = best_row_for_neighbors, best_col_for_neighbors


                
        
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

def accuracy_block_base(original_img, final_img, rows, cols, patch_width, patch_height):
    correct_blocks = 0
    total_blocks = rows * cols
    for row in range(rows):
        for col in range(cols):
            patch = final_img[row*patch_height:(row+1)*patch_height, col*patch_width:(col+1)*patch_width]
            original_patch = original_img[row*patch_height:(row+1)*patch_height, col*patch_width:(col+1)*patch_width]
            if np.array_equal(patch, original_patch):
                correct_blocks += 1
    return 100 * correct_blocks / total_blocks


for puzzle_idx in range(start_puzzle, end_puzzle + 1):
    for slot in slot_types:
        folder_path = os.path.join(base_data_path, f"Puzzle_{puzzle_idx}_{slot}")
        
        if not os.path.exists(folder_path):
            print(f"Skipping {folder_path} (not found)")
            continue

        output_file = os.path.join(folder_path, "Output.tif")
        patch_files = glob.glob(os.path.join(folder_path, "Patch_*.tif"))
        if not patch_files or not os.path.exists(output_file):
            print(f"Skipping {folder_path} (missing files)")
            continue

        corners_img = np.array(Image.open(output_file))
        base_img = corners_img.copy()
        patches = [np.array(Image.open(pf)) for pf in patch_files]

        rows, cols = base_img.shape[0] // patches[0].shape[0], base_img.shape[1] // patches[0].shape[1]
        img_width, img_height = 1920, 1200
        patch_width, patch_height = img_width // cols, img_height // rows

        print(f"Solving Puzzle_{puzzle_idx}_{slot} - {rows}x{cols}, {len(patches)} patches")

        final_img = puzzle_solver(base_img, patches)
        Image.fromarray(final_img).save(os.path.join(folder_path, "final_solved.tif"))

        original_img = np.array(Image.open(os.path.join(folder_path, "Original.tif")))
        accuracy = accuracy_block_base(original_img, final_img, rows, cols, patch_width, patch_height)
        print(f"Puzzle_{puzzle_idx}_{slot} Accuracy: {accuracy:.2f}%")
