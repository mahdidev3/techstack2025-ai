import cv2

def sharpen_image(image, kernel_size=(7, 7), gamma=2):

  blurred_image = cv2.GaussianBlur(image, kernel_size , 10)

  sharpen_image = cv2.subtract(image, blurred_image)

  sharpen_image = cv2.multiply(sharpen_image, gamma)
  
  sharpened_image = cv2.add(image, sharpen_image)

  sharpened_image = cv2.convertScaleAbs(sharpened_image)
  return sharpened_image, blurred_image


image = cv2.imread('./data/tiger.jpg')
if image is not None:
  sharpened_img, blur = sharpen_image(image)
  cv2.imshow("Original Image", image)
  cv2.imshow("Sharpened Image", sharpened_img)
  cv2.imshow("Blurred Image", blur)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
else:
  print("Error: Could not load image.")