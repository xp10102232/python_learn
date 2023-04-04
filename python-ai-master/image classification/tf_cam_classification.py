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

if __name__ == "__main__":
    # 输出概率最大的三个分类结果
    Top_K = 3




    # 分类模型
    file_model = "mobileNet_V1\\mobilenet_v1_1.0_224_quant.tflite"

    # 标签列表
    file_label = "mobileNet_V1\\labels_mobilenet_quant_v1_224_cn_baidu.txt"

    # 读取标签
    with open(file_label, 'r',encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
      
    # 加载分类模型
    interpreter = tflite.Interpreter(model_path=file_model)
    interpreter.allocate_tensors()

    # 读取输入数据细节
    input_details = interpreter.get_input_details()
    print('Info of input')
    print(input_details)

    # 读取输出数据的细节
    output_details = interpreter.get_output_details()
    print('Info of output')
    print(output_details)

    # 获取输入图像的尺寸要求
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]


    # 打开摄像头
    url = "http://admin:admin@192.168.3.27:8081"
    # url = 0
    cap = cv2.VideoCapture(url)

    # 初始化帧率计算
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    while True:

        # 获取起始时间
        t1 = cv2.getTickCount()
        
        # 读取一帧图像
        success, img = cap.read()

        # 获取它的尺寸
        imH,imW,_ = np.shape(img)

        # BGR 转RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 尺寸缩放适应网络输入要求
        img_resized = cv2.resize(img_rgb, (width, height))

        # 维度扩张适应网络输入要求
        input_data = np.expand_dims(img_resized, axis=0)

        # 数据输入网络
        interpreter.set_tensor(input_details[0]['index'],input_data)

        # 进行识别
        interpreter.invoke()

        # 获得输出
        outputs = interpreter.get_tensor(output_details[0]['index'])[0] 
        output = np.squeeze(outputs)

        # 根据量化情况对输出进行还原
        if output_details[0]['dtype'] == np.uint8:
            scale, zero_point = output_details[0]['quantization']
            output = scale * (output - zero_point)

        # 找到Top-K 个最大值    
        ordered = np.argpartition(-output, Top_K-1)    

        # 输出标签以及分类的概率输出
        for i in range(Top_K):
            str_info = "%s %.2f%%"%(labels[ordered[i]],output[ordered[i]]*100)
            pos = (1,1+i*25)
            img = paint_chinese_opencv(img,str_info,pos,(255,0,0))
            
        cv2.putText(img,'FPS: %.2f'%(frame_rate_calc),(imW-200,imH-20),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        
        # 显示结果
        cv2.imshow('Result', img)

        # 计算帧率
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        
         
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
    cap.release() 













