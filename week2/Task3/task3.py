import cv2

# Load image (grayscale)
img = cv2.imread("./data/bacteria.jpg", cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

binary = cv2.bitwise_not(binary)

eroded_kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (35,35))

eroded = cv2.erode(binary , eroded_kernel , iterations=1)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)

print(f"Number of components (excluding background): {num_labels - 1}")

gray_output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
eroded_output = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    cv2.rectangle(gray_output, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(gray_output, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    
    cv2.rectangle(eroded_output, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(eroded_output, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imshow("Original", img)
cv2.imshow("Binary", binary)
cv2.imshow("Components on gray", gray_output)
cv2.imshow("Components on enroded", eroded_output)


cv2.waitKey(0)
cv2.destroyAllWindows()