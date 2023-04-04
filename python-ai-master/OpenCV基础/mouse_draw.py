import cv2
import numpy as np

def points_collect(event,x,y,flags,param):
    dic_points = param
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if len(dic_points['p2'])>0:
            dic_points['p2']=()
            dic_points['p1']=(x,y)
        elif len(dic_points['p1'])==0:
            dic_points['p1']=(x,y)
        elif len(dic_points['p1'])>0:
            dic_points['p2']=(x,y)
            
    if event == cv2.EVENT_MOUSEMOVE:
        dic_points['p_move']=(x,y)
        
        
def drawline(img,dic_points):
    color = (0,255,255)
    if len(dic_points['p2'])>0:
        cv2.circle(img,dic_points['p1'],2,color,cv2.FILLED)
        cv2.circle(img,dic_points['p2'],2,color,cv2.FILLED)
        cv2.line(img,dic_points['p1'],dic_points['p2'],color,1)
    elif len(dic_points['p1'])>0 and len(dic_points['p_move'])>0:
        cv2.circle(img,dic_points['p1'],2,color,cv2.FILLED)
        cv2.line(img,dic_points['p1'],dic_points['p_move'],color,1)

# 图像叠加
def img_overlayer(img,img_fg,pos_fg,bk_fg):
    
    #把前景图变换为灰度
    fg_gray = cv2.cvtColor(img_fg,cv2.COLOR_BGR2GRAY)
    h_gf,w_fg = np.shape(fg_gray)
    
    
    # 获取前景图的mask 有图部分为 1 背景部分为 0 
    if bk_fg == 255:
        mask_fg = fg_gray<250
    elif bk_fg == 0:
        mask_fg = fg_gray>5
    
    mask_fg = mask_fg[:,:,np.newaxis]
    not_mask_fg = ~mask_fg
    
    # 截取背景图
    bk = img[pos_fg[1]:pos_fg[1]+h_gf,pos_fg[0]:pos_fg[0]+w_fg]
    
    img_overlayer = bk*not_mask_fg + img_fg*mask_fg
    img[pos_fg[1]:pos_fg[1]+h_gf,pos_fg[0]:pos_fg[0]+w_fg] = img_overlayer
    return img
    
def add_glass(img,img_glass,dic_points,bk_fg=255):
    
    if len(dic_points['p2'])==0:
        return img
    
    # 获取眼镜图像大小
    w_glass = np.shape(img_glass)[1]
    h_glass = np.shape(img_glass)[0]
    
    # 计算缩放尺度
    scale = np.abs(dic_points['p1'][0]-dic_points['p2'][0])/w_glass
    
    # 眼镜图像缩放
    resize_glass = cv2.resize(img_glass,(int(w_glass*scale),int(h_glass*scale)))

    # 计算眼镜图像的起始位置(左上坐标)
    pos_glass = (dic_points['p1'][0],dic_points['p1'][1]-int(h_glass*scale/2.0))
    
    # 图像叠加
    img_out = img_overlayer(img,resize_glass,pos_glass,bk_fg)
    
    return img_out
    
  
if __name__=="__main__":
    
    # 记录坐标的字典
    dic_points = {}
    dic_points["p1"]=()
    dic_points["p2"]=()
    dic_points["p_move"]=()
    
    # 设置回调函数
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', points_collect,param=dic_points)
    
    # 不加眼睛
    flag_add_glass = 0
    
    # 读取眼睛图像
    img_glass = cv2.imread("glasses.bmp")
    
    while True:
        img = cv2.imread("cat.bmp")  #加载图片 
        
        if flag_add_glass ==0:
            drawline(img,dic_points)
        
        if flag_add_glass ==1:
            img = add_glass(img,img_glass,dic_points)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        if key == ord('a'):
           flag_add_glass = 1
           key =0
        if key == ord('r'):
           flag_add_glass = 0
           key =0
        
        cv2.imshow("image",img)
        
        
