import cv2

img = cv2.imread("./data/bacteria.jpg", cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

dilated = cv2.dilate(binary, kernel, iterations=1)
eroded = cv2.erode(binary, kernel, iterations=1)
edges = cv2.absdiff(dilated, eroded)


dilated_edges =  cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)), iterations=1)
cleaned_edges = cv2.erode(dilated_edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)

cv2.imshow("Eroded" , eroded)
cv2.imshow("Dilated" , dilated)
cv2.imshow("Original", img)
cv2.imshow("Binary", binary)
cv2.imshow("Edges", edges)
cv2.imshow("Cleaned Edges", cleaned_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()