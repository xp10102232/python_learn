import cv2
import numpy as np
if __name__ == "__main__":
    # 读取图像并显示
    img= cv2.imread("cat.bmp")
    print("image size",np.shape(img))
    cv2.imshow("output",img)


    # 图像缩放
    h = np.shape(img)[0]
    w = np.shape(img)[1]
    scale = 2

    imgResize1 = cv2.resize(img,(int(w*scale),int(h*scale)))
    cv2.imshow("reszie1",imgResize1)
    print("imresize1 size ",np.shape(imgResize1))
    
    scale = 0.5
    imgResize2 = cv2.resize(img,(int(w*scale),int(h*scale)))
    cv2.imshow("reszie2",imgResize2)
    print("imgresize2 size",np.shape(imgResize2))
    
    #图像剪裁
    imgCropped = img[int(h/3):int(2*h/3),int(w/3):int(w*2/3)]
    cv2.imshow("cropped",imgCropped)
    print("imgcropped size",np.shape(imgCropped))
    
    # 颜色变换
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print("image_Gray size",np.shape(img_Gray))
    cv2.imshow("RGB",img_RGB)
    cv2.imshow("Gray",img_Gray)
    
    # 等待按键
    cv2.waitKey(0)