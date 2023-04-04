import cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw

def paint_chinese_opencv(im,chinese,pos,color,font_size = 20):
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('NotoSansCJK-Bold.ttc',font_size,encoding="utf-8")
    fillColor = color 
    position = pos 
   
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,fillColor,font)
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    return img

if __name__=="__main__":
    # 创建一个纯黑的图像用来进行绘图展示
    img = np.zeros((512,512,3),np.uint8)

    '''画直线'''
    # 点1 坐标
    p1 = (0,0)
    # 点2 坐标
    p2 = (img.shape[1],img.shape[0])
    # 线的颜色
    color = (0,255,0)
    # 线的宽度
    size_line = 3
    cv2.line(img,p1,p2,color,size_line)

    '''画矩形 空心'''
    # 矩形左上角坐标
    pos1_rect = (0,0)
    # 矩形的右下角坐标
    pos2_rect  = (250,350)
    # 矩形颜色
    color = (0,0,255)
    # 线宽
    size_line = 2 
    cv2.rectangle(img,pos1_rect,pos2_rect,color,size_line)

    '''画矩形 实心'''
    cv2.rectangle(img,(100,100),(200,200),(255,0,0),cv2.FILLED )

    '''画圆形 空心'''
    # 圆心
    p_center = (400,50)
    # 半径
    len_R = 30
    cv2.circle(img,p_center,len_R,(255,255,0),5)

    '''画圆形 实心'''
    cv2.circle(img,(450,80),30,(0,255,255),cv2.FILLED)

    '''文字输出'''
    # 输出的文字
    str_txt = "OpenCV"
    
    # 文字输出区域的左上角坐标
    pos_txt = (300,200)
    
    # 字体
    font = cv2.FONT_HERSHEY_COMPLEX
    
    # 字号
    font_size =1
    
    # 颜色
    color = (0,150,0)
    
    # 绘制文字的线宽
    line_size = 3
    
    cv2.putText(img,str_txt,pos_txt,font,font_size,color,line_size)

    '''中文输出'''
    color_rgb = (150,0,0)
    str_txt = "这是中文"
    pos_text = (300,250)
    img = paint_chinese_opencv(img,"这是中文",pos_text,color_rgb)

    cv2.imshow("Image",img)
    cv2.waitKey(0)