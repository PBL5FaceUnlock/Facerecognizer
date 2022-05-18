# Main functions using for recognization of APIs
import cv2
import os
from facenet_pytorch import MTCNN
import torch
from array import array
from global_config import PROCESSED_IMAGE_DIR_SUFFIX, RAW_IMAGE_DIR_SUFFIX
from detect import get_cropped_images
from utils import crop_and_save
from recognize import get_recognized_person_info, save_feature_vectors
import warnings
warnings.filterwarnings('ignore')


def recognize_students_in_image(img_path: str):
    # recognize students
    # params:
    #   img_path: path of image
    # return: list of person name
    cropped_face_list: array = get_cropped_images(img_path)
    recog_faces: array = []
    for index, cropped_item in enumerate(cropped_face_list):
        print(f"{index}")
        person_name = get_recognized_person_info(cropped_item, index)
        if person_name != "":
            recog_faces.append(person_name)
    return recog_faces

#######train data
def init_atribute_vectors():
    # init attribute vectors from images in dataset

    current_path = str(os.path.abspath(os.getcwd()))  # pbl5-api

    raw_image_dir = current_path + RAW_IMAGE_DIR_SUFFIX
    processed_image_dir = current_path + PROCESSED_IMAGE_DIR_SUFFIX
    sub_folders = [
        name for name in os.listdir(raw_image_dir)
        if os.path.isdir(os.path.join(raw_image_dir, name))
    ]

    for sub_folder in sub_folders:
        raw_path = raw_image_dir + "/" + sub_folder
        processed_path = processed_image_dir + "/" + sub_folder
        crop_and_save(raw_path, sub_folder)
    save_feature_vectors()

def get_image_data():
    # chup 30 anh khi da detect = MTCNN
    device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

    cap = cv2.VideoCapture(0) 
    name = input("Enter your Name:")
    sampleNum = 1
    path= 'Dataset/raw'
    userNames=os.listdir(path)
    if name not in userNames:
        os.makedirs('Dataset/raw/'+name)
    while (True):
        ret, frame = cap.read()
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                bbox = list(map(int,box.tolist()))
                try:
                    cv2.imwrite('Dataset/raw/'+ name +'/pic.' + str(sampleNum) + '.jpg',frame)
                    cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
                    print("Chup anh thu " + str(sampleNum))
                    sampleNum += 1
                except Exception as e:
                    print('Co loi!')
        
        cv2.imshow('frame',frame)
        cv2.waitKey(100)
        if sampleNum > 30:                 # thoat neu du 30 anh
            break
    cap.release()
    cv2.destroyAllWindows()




# get_image_data()

# init_atribute_vectors()

print(recognize_students_in_image("D:\\WORK-SCH\\6\\Machine-Learning\\facerecognize\\nhat-test.jpg"))
