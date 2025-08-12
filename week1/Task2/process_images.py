import os
import cv2
import numpy as np
from pathlib import Path

def create_puzzle_dirs(raw_data_path, data_path, puzzle_index, slot_configs=[(40, 240, 240), (160, 120, 120)]):
    raw_data_path = Path(raw_data_path)
    data_path = Path(data_path)
    
    for img_path in raw_data_path.glob('*'):
        if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.png', '.tif', '.tiff']:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
                
            # Check and log base image dimensions
            h, w = img.shape[:2]
            expected_dims = (1920, 1200)
            print(f"Processing {img_path.name}: Dimensions are {w}x{h} "
                  f"{'(matches expected)' if (w, h) == expected_dims else '(does NOT match expected 1920x1200)'}")
            
            for slots, patch_w, patch_h in slot_configs:
                rows = int(h / patch_h)
                cols = int(w / patch_w)
                if rows * cols != slots:
                    print(f"Warning: {slots} is not a perfect square, skipping...")
                    continue
                
                puzzle_dir = data_path / f"Puzzle_{puzzle_index}_{slots}"
                puzzle_dir.mkdir(exist_ok=True)
                
                # Save original
                cv2.imwrite(str(puzzle_dir / 'Original.tif'), img)
                
                patches = []
                patch_num = 1  # Continuous numbering for non-corner patches
                for i in range(rows):
                    for j in range(cols):
                        # Skip the four corners
                        if (i, j) in [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]:
                            continue
                        patch = img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                        cv2.imwrite(str(puzzle_dir / f'Patch_{patch_num}.tif'), patch)
                        patches.append(patch)
                        patch_num += 1
                
                # Save corners separately
                corners = [
                    (1, 1, img[:patch_h, :patch_w]),
                    (1, cols, img[:patch_h, -patch_w:]),
                    (rows, 1, img[-patch_h:, :patch_w]),
                    (rows, cols, img[-patch_h:, -patch_w:])
                ]
                for row, col, corner in corners:
                    cv2.imwrite(str(puzzle_dir / f'Corner_{row}_{col}.tif'), corner)
                
                # Create shuffled patches image (without corners)
                np.random.shuffle(patches)
                shuffled = np.zeros_like(img)
                
                patch_idx = 0
                for i in range(rows):
                    for j in range(cols):
                        if (i, j) in [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]:
                            shuffled[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                        else:
                            shuffled[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = patches[patch_idx]
                            patch_idx += 1
                
                cv2.imwrite(str(puzzle_dir / 'Shuffled_Patches.tif'), shuffled)
                
                # Save final
                cv2.imwrite(str(puzzle_dir / 'final.tif'), img)
                
                # Create output image with only corners in place
                output = np.zeros_like(img)
                output[:patch_h, :patch_w] = img[:patch_h, :patch_w]  # Top-left
                output[:patch_h, -patch_w:] = img[:patch_h, -patch_w:]  # Top-right
                output[-patch_h:, :patch_w] = img[-patch_h:, :patch_w]  # Bottom-left
                output[-patch_h:, -patch_w:] = img[-patch_h:, -patch_w:]  # Bottom-right
                cv2.imwrite(str(puzzle_dir / 'Output.tif'), output)
                puzzle_index += 1

if __name__ == "__main__":
    raw_data_path = r"C:\Users\mahdi\OneDrive\Desktop\TechStack\Tasks\techstack2025-ai\week1\Task2\row data"
    data_path = r"C:\Users\mahdi\OneDrive\Desktop\TechStack\Tasks\techstack2025-ai\week1\Task2\data"
    create_puzzle_dirs(raw_data_path, data_path, 3)
