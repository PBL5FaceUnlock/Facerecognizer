import os
import numpy as np
from cv2 import cv2
from mtcnn import MTCNN
import warnings
warnings.filterwarnings('ignore')



def get_cropped_images(image_path):
    detector = MTCNN()
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    cap = cv2.imread(image_path)

    bounding_boxes_list = []

    for i in result:
        bounding_boxes_list.append(i["box"])

    bounding_boxes = np.asfarray(bounding_boxes_list)
    faces_found = bounding_boxes.shape[0]
    print(str(faces_found) + " nguoi")
    det = bounding_boxes[:, 0:4]
    bb = np.zeros((faces_found, 4), dtype=np.int32)

    cropped_list = []

    image_index = 0
    for i in range(faces_found):
        bb[i][0] = det[i][0]
        bb[i][1] = det[i][1]
        bb[i][2] = det[i][2]
        bb[i][3] = det[i][3]

        x = bb[i][0]
        y = bb[i][1]
        w = bb[i][2]
        h = bb[i][3]
        cropped = cap[y:y + h, x:x + w]
        if cropped.size != 0:
            crop_img2 = cv2.resize(cropped, (160, 160))
            img_name = str(image_index + 1) + ".jpg"

            #  This line's used for debug
            current_path = str(os.path.abspath(os.getcwd()))  # .../pbl5-api
            cv2.imwrite(
                os.path.join(current_path + "/cropped/", img_name), crop_img2
            )
            cv2.waitKey(0)
            image_index = image_index + 1
            cropped_list.append(cropped)

    return cropped_list
