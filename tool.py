import cv2
import os

if __name__ == "__main__":
    root_path = "data/crop"
    images = os.listdir(root_path)
    for img in images:
        img_path = os.path.join(root_path, img)
        image = cv2.imread(img_path)

        new_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

        cv2.imwrite(
            os.path.join(root_path, img.replace(".jpg", "_resize.jpg")), new_image
        )
