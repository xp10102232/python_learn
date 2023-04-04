import cv2
import numpy as np
from knnDetector import knnDetector
from sort import Sort
import adjuster
from collections import deque
from datetime import datetime
import os
stat_dic= {0:'null',
           1: 'cap_start',
           2: 'caping',
           3:'cap_waite',
           4: 'cap_stop'
           }
# 视频保存           
def save_video(datas,fps):
    # 图片保存
    currentDateAndTime = datetime.now()
    floder_name = currentDateAndTime.strftime("%Y-%m-%d")
    video_name = currentDateAndTime.strftime("%Y-%m-%d-%H-%M-%S")+".avi"
    os.makedirs(os.path.join('videos',floder_name),exist_ok=True)
    wav_save = os.path.join('videos',floder_name,video_name)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  #XVID
    h,w,_ = datas[0].shape
    out = cv2.VideoWriter(wav_save, fourcc, fps, (w,h))
    for d in datas:
        out.write(d)
    return out


def det_cap(path,b_adjust=False):
    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 200)
    fps = capture.get(cv2.CAP_PROP_FPS)
  

    # 定义检测器
    detector = knnDetector(500, 400, 10)

    # 定义目标跟踪器
    sort = Sort(3, 5, 0.1)
   

    # 定义标志位
    flag = False

    # 读取一帧，作为背景模板
    ret, frame = capture.read()

    # 对图像进行预处理进行简单的裁剪
    adjust = adjuster.Adjuster(frame, (120, 60))
    state = 0
    state_pass = -1
    n_wait= 0
    n_frame =0

    # 定义存储前端
    catch_front= deque(maxlen=20)
    video_save =[]
    while True:
        tracked = False

        ret, frame = capture.read()
        if frame is None:
            break
        n_frame = n_frame + 1
        # 将当前帧和存储的背景帧进行对齐
        if b_adjust:
            frame = adjust.debouncing(frame)
        catch_front.append(frame)

        # 进行目标检测
        mask, bboxs = detector.detectOneFrame(frame)

        if bboxs != []:
            bboxs = np.array(bboxs)
            bboxs[:, 2:4] += bboxs[:, 0:2]
           
            trackBox = sort.update(bboxs)
        else:
            # test
            trackBox = sort.update()

        # 有跟踪目标则进行绘图
        for bbox in trackBox:

            bbox = [int(bbox[i]) for i in range(5)]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
            cv2.putText(frame, str(bbox[4]), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            tracked = True
        
        # 如果没有开始录制,但检测到跟踪目标则，开始进行录制
        if state ==0 and tracked == True:
            state = 1 
            n_wait =0
        # 如果开始录制，检测到跟踪目标，则进入正在录制状态
        elif state ==1 and  tracked == True:
            state = 2
            n_wait =0
        
        # 正在录制状态下，没有跟踪到目标则进入等待状态    
        elif state ==2 and tracked == False:
            state =3
            n_wait = n_wait+1
        
        # 等待状态下，且没有跟踪到目标，等待时间 +1
        elif state ==3 and tracked == False:
            n_wait = n_wait+1
            # 如果没有跟踪上的时间过长则停止跟踪
            if n_wait >30:
                state = 4
        
        # 如果等待状态下 又跟踪到目标，则进入正在录制状态        
        elif state ==3 and tracked == True:
            state =2
            n_wait =0

        elif state == 4 and tracked == False:
            state =0


        if not state_pass == state:
            print("%d %s"%(n_frame,stat_dic[state]))
        
        state_pass = state


        if state == 1:
            video_save = [d for d in catch_front]
            out = save_video(video_save,fps)
        elif state ==2 or state == 3:
            out.write(frame)
        elif state ==4:
            print("保存视频")
            out.release()
            

        cv2.imshow("frame", frame)

         # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    capture.release() 





if __name__ == "__main__":
    det_cap('IMG_4550.MOV',False)


  





