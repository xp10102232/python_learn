
import numpy as np
from  kalman_filter import KalmanFilter
import cv2
from detecter import infer_image,plot_one_box
import time

# 计算iou
def bbox_iou(box1, box2):
        
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[ 0], box1[ 1], box1[ 2], box1[ 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[ 2], box2[3]

      
    # 获取重叠部分的坐标
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        
    # 计算重叠部分的面积
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                    np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
    
    # 计算iou 重叠面积的比例
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

# x1y1x2y2 转 xyah
def tlbr_to_xyah(tlbr):
    ret = np.asarray(tlbr).copy()
    ret[2:] -= ret[:2]
    
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    
    return ret
    
    
def xyah_to_tlbr(xyah):
    ret = np.asarray(xyah).copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    
    ret[2] = ret[0] + ret[2]
    ret[3] = ret[1] + ret[3]
    return ret
    
 
if __name__ == "__main__":
    
    # 进行模型加载
    dic_labels= {0:'led',
            1:'buzzer',
            2:'teeth'}

    model_h = 640
    model_w = 640
    file_model = 'best-led-640.onnx'
    net = cv2.dnn.readNet(file_model)
    
    # 定义跟踪状态
    mean_tracking = None
    covariance_tracking = None
    state = False
    n_track_error =0
    
    # 定义卡尔曼滤波器
    m_filter = KalmanFilter()
    
    # 定义视频源
    video = 0
    cap = cv2.VideoCapture(video)
    
    
    while True:
        success, img0 = cap.read()
        if success:
            
            det_box =[]
            mean_show = []
            det_last= []
            
            # 进行目标检测
            t1 = time.time()
            det_boxes,scores,ids = infer_image(net,img0,model_h,model_w,thred_nms=0.4,thred_cond=0.5)
            t2 = time.time()
            
            # 找到蓝牙音箱的位置
            id_teeth = np.where(ids == 2)[0]
            
            if len(id_teeth)>0:
                det_box = det_boxes[id_teeth[0]]
                if len(det_box.shape)==2:
                    det_box = det_boxes[id_teeth[0]][0]
                
                # 如果没有开始跟踪，则进行卡尔曼滤波器的初始化
                if not state:
                    mean_tracking,covariance_tracking = m_filter.initiate(tlbr_to_xyah(det_box))
                    state = True
                # 如果已经开始跟踪 进行参数的更新
                else:
                    # 更新
                    mean_tracking,covariance_tracking = m_filter.update(mean_tracking,covariance_tracking,tlbr_to_xyah(det_box))
                    # 预测
                    mean_tracking,covariance_tracking = m_filter.predict(mean_tracking,covariance_tracking)
                mean_show = mean_tracking.copy()
                last_box = det_box.copy()
           
            # 如果没有检测到目标
            else:
                # 但是已经开启跟踪了
                if state:
                    # 进行预测
                    mean_tracking,covariance_tracking= m_filter.predict(mean_tracking,covariance_tracking)
                    # mean_tracking,covariance_tracking = m_filter.update(mean_tracking,covariance_tracking, xyah_to_tlbr(mean_tracking[:4]))
                    mean_show = mean_tracking.copy()
                    # # 利用预测值进行参数更新
                    # mean_tracking,covariance_tracking = m_filter.update(mean_tracking,covariance_tracking,mean_predict[:4])
                    
                    # mean_show,_ = m_filter.predict(mean_tracking,covariance_tracking)
                    
            
            if len(det_box)>0:
                # print(det_box)
                plot_one_box(det_box.astype(np.int16), img0, color=(255,0,0), label="det", line_thickness=None)
            
            if len(mean_show)>0:
                print(mean_show)
                box_track = xyah_to_tlbr(mean_show[:4])
                plot_one_box(box_track.astype(np.int16), img0, color=(0,0,255), label="track", line_thickness=None)
                
                # 计算iou
                iou = bbox_iou(last_box,box_track)
                if iou<0.3:
                    n_track_error= n_track_error +1
                    if n_track_error>8:
                        state = False
                else:
                    n_track_error = 0







                
            # str_FPS = "FPS: %.2f"%(1./(t2-t1))
            
            # cv2.putText(img0,str_FPS,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
            
            # if len(det_box)>0 and len(mean_show)>0:
            #     box_track = xyah_to_tlbr(mean_show[:4])
            #     iou = bbox_iou(det_box,box_track)
            #     print("iou %.2f"%(iou))
            #     # iou 过小 则 重新初始化滤波器
            #     if iou<0.3:
            #         mean_tracking,covariance_tracking = m_filter.initiate(tlbr_to_xyah(det_box))


            cv2.imshow("video",img0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release() 
    




















    
    






# class Tracker()
    # def __init__(self, x1y1x2y2):
        # self.kalman_filter = None
        # self.mean, self.covariance = None, None
        # self.is_activated = False

        # self.score = score
        # self.tracklet_len = 0
    
    # def predict(self):
        # mean_state = self.mean.copy()
        # if self.state != TrackState.Tracked:
            # mean_state[7] = 0
        # self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
    
    # def 