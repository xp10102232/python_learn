import cv2
import numpy as np
import pytesseract

# 扫描文件四个顶点的收集程序
# 双击进行收集
def points_collect(event,x,y,flags,param):
    dic_points = param
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if len(dic_points['ps'])>=4:
            dic_points['ps']=[]
            dic_points['ps'].append((x,y))
        else:
            dic_points['ps'].append((x,y))
            
    if event == cv2.EVENT_MOUSEMOVE:
        dic_points['p_move']=(x,y)

def drawlines(img,dic_points):
    color = (0,255,0)
    
    # 已记录顶点复制
    points = dic_points['ps'][:]
    #追加移动动点
    points.append(dic_points['p_move'])
    if len(points)>0 and len(points)<5:
        for i in range(len(points)-1):
            cv2.circle(img,points[i],4,color,cv2.FILLED)
            cv2.line(img,points[i],points[i+1],color,1)
    elif len(points)>=5:
        for i in range(3):
            cv2.circle(img,points[i],4,color,cv2.FILLED)
            cv2.line(img,points[i],points[i+1],color,1)
        
        cv2.circle(img,points[3],4,color,cv2.FILLED)
        cv2.line(img,points[3],points[0],color,1)

# 将收集的四个顶点按照[左上，右上，左下，右下]
# 顺序进行重新排列        
def reorder(points):
    points = np.array(points)
    ordered_points = np.zeros([4,2])
    
    # 将横纵坐标相加，
    # 最小为左上角，最大为右下角 
    add = np.sum(points,axis=1) 
    ordered_points[0] = points[np.argmin(add)]
    ordered_points[3] = points[np.argmax(add)]
    
    # 将横纵坐标相减 diff 为后减前 即 y-x
    # 最小为右上角，最大为左下角
    diff = np.diff(points,axis=1)
    ordered_points[1] = points[np.argmin(diff)]
    ordered_points[2] = points[np.argmax(diff)]
    return ordered_points

# 实现图像的仿射变换
# ordered_points ： 需要变换的4个顶点
# size_wraped: 变换后 图像的大小 （w,h）    
def getWarp(img,ordered_points,size_wraped):
    w,h = size_wraped
    
    # 源图像坐标点
    ps1 = np.float32(ordered_points)
    
    # 目标图像坐标点
    ps2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    
    # 计算仿射矩阵
    matrix = cv2.getPerspectiveTransform(ps1, ps2)
    
    # 进行仿射变换
    imgOutput = cv2.warpPerspective(img, matrix, (w, h))
    
    # 对边界进行简单裁剪
    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(w,h))
    return imgCropped
    


         
if __name__ == "__main__":
    
    # 记录坐标的字典
    dic_points = {}
    dic_points["ps"]=[]
    dic_points["p_move"]=()
    
    # 设置回调函数
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', points_collect,param=dic_points)
    
    # 需要扫描的文件
    file_scan = "paper.jpg"
    
    # 扫描文件的大小
    size_wraped = (420,600)
    
    # 定义tesseract的位置
    pytesseract.pytesseract.tesseract_cmd = 'D:\\工作相关\\我设计的课程\\python与人工智能课程设计\\应用篇\\文字扫描仪\\tesseract\\tesseract.exe'

    while True:
        img = cv2.imread(file_scan)
        
        drawlines(img,dic_points)
        
        key=cv2.waitKey(10) & 0xFF 
        cv2.imshow('image',img)
        
        if key == ord('q'):
            break
            
        if key == ord('w'):
            key = 0
            if len(dic_points['ps'])==4:
                
                # 图像仿射变换
                ordered_points = reorder(dic_points['ps'])
                img_Warped = getWarp(img,ordered_points,size_wraped)
                cv2.imshow("ImageWarped",img_Warped)
                
                # 颜色转换
                imgWarped_RGB = cv2.cvtColor(img_Warped, cv2.COLOR_BGR2RGB)
                
                # 文字识别
                txt = pytesseract.image_to_string(imgWarped_RGB, lang='chi_sim')
                print(txt)
    
   