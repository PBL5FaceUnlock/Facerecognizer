import os

from cv2 import cv2
from mtcnn import MTCNN
from global_config import PROCESSED_IMAGE_DIR_SUFFIX
import warnings
warnings.filterwarnings('ignore')



def crop_and_save(raw_path, name):
    images = []
    dem = 0
    current_path = str(os.path.abspath(os.getcwd()))  # .../pbl5-api
    newpath = current_path + PROCESSED_IMAGE_DIR_SUFFIX + "/" + name
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    for filename in os.listdir(raw_path):
        # print(filename)
        img = cv2.imread(os.path.join(raw_path, filename))
        if img is not None:
            images.append(img)
            detector = MTCNN()
            result = detector.detect_faces(img)
            for item in result:
                x = item['box'][0]
                y = item['box'][1]
                w = item['box'][2]
                h = item['box'][3]
                crop_img = img[y:y + h, x:x + w]

                if crop_img.size == 0:
                    continue

                crop_img2 = cv2.resize(crop_img, (160, 160))
                img_name = str(dem + 1) + ".jpg"
                cv2.imwrite(os.path.join(newpath, img_name), crop_img2)
                cv2.waitKey(0)
                dem = dem + 1
