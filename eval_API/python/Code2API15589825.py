import cv2

def crop_image(img_path, x, y, w, h):
    img = cv2.imread(img_path)
    crop_img = img[y:y+h, x:x+w]
    return crop_img
