import cv2
import numpy as np

# Load image (grayscale)
img = cv2.imread("./data/bacteria.jpg", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binary = cv2.bitwise_not(binary)
print(_)
morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (5,5))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel)
# binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel)

sure_bg = cv2.erode(binary, morph_kernel, iterations=4)

dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)

print(f"Number of components (excluding background): {num_labels - 1}")

gray_output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
binery_output = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR)

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    cv2.rectangle(gray_output, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(gray_output, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    
    cv2.rectangle(binery_output, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(binery_output, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imshow("Original", img)
cv2.imshow("Binary", binary)
cv2.imshow("Components on gray", gray_output)
cv2.imshow("Components on binery", binery_output)


cv2.waitKey(0)
cv2.destroyAllWindows()