import cv2
from source.cartoonize import Cartoonizer
import os
import numpy as np


def get_model_list(model_dir):
    list_models = []
    m_dirs = os.listdir(model_dir)
    for dir in m_dirs:
        path_model = os.path.join(model_dir,dir)
        list_models.append(path_model)
    return list_models


if __name__ == '__main__':
    

    i=4

    list_models = get_model_list("models")
    algo =Cartoonizer(list_models[i])

    cap = cv2.VideoCapture(0)
    while True:
        success,img = cap.read()
        if  success:
            result = algo.cartoonize(img[...,::-1])

            cv2.imshow('video',np.array(result,dtype=np.uint8))
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release() 



    
    # img = cv2.imread(file_img)[...,::-1]
    # result = algo.cartoonize(img)
    # result_out = np.array(result,dtype=np.uint8)
    # # cv2.namedWindow("out", cv2.WINDOW_NORMAL or cv2.WINDOW_KEEPRATIO or cv2.WINDOW_GUI_NORMAL)
    # # cv2.imshow("input",cv2.imread('input.png'))       
    # # cv2.imshow("out",result)
    # # cv2.waitKey(0)
    # file_img_out = os.path.split(list_models[i])[-1]+"_out_"+file_img
    # cv2.imwrite(file_img_out,result_out)