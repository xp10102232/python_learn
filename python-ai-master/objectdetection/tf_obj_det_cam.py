import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from PIL import Image,ImageFont,ImageDraw

def paint_chinese_opencv(im,chinese,pos,color):
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('NotoSansCJK-Bold.ttc',25,encoding="utf-8")
    fillColor = color #(255,0,0)
    position = pos #(100,100)
    # if not isinstance(chinese,unicode):
        # chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,fillColor,font)
 
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    return img

if __name__=="__main__":
    
    # 设置检测阈值
    min_conf_threshold  = 0.35

    # 检测模型
    file_model = "model_obj_detect\\detect.tflite"
    # 标签
    file_label = "model_obj_detect\\labelmap_cn.txt"

    # 获取标签
    with open(file_label, 'r',encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]

    if labels[0] == '???':
        del(labels[0])  
        
    # 载入模型
    interpreter = tflite.Interpreter(model_path=file_model)
    interpreter.allocate_tensors()

    # 获取输入、输出的数据的信息
    input_details = interpreter.get_input_details()
    print('input_details\n',input_details)
    output_details = interpreter.get_output_details()
    print('output_details',output_details)

  
    # 获取输入图像的高和宽
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # 打开摄像头
    url = "http://admin:admin@192.168.3.27:8081"
    cap = cv2.VideoCapture(url)

    # 初始化帧率计算
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

   
    while True:

        # 获取起始时间
        t1 = cv2.getTickCount()
        
        # 读取一帧图像
        success, frame = cap.read()
        
        # 获取图像的宽和高
        imH,imW,_ = np.shape(frame)

        # RGB 转 BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # 输入图像    
        interpreter.set_tensor(input_details[0]['index'],input_data)
        
        # 进行检测
        interpreter.invoke()

        # 获取检测结果
        # 检测物体的边框
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        # 检测物体的类别
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        
        # 检测物体的分数
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

        # 对于概率大于 50%的进行显示
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                
                # 获取边框坐标
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                # 画框
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # 获取检测标签
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                
                #显示标记
                frame = paint_chinese_opencv(frame,label,(xmin,ymin-5),(255,0,0))
        
        # 显示帧率
        cv2.putText(frame,'FPS: %.2f'%(frame_rate_calc),(imW-200,imH-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        # 显示结果
        cv2.imshow('object detect', frame)

        # 计算帧率
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        
         # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release() 







