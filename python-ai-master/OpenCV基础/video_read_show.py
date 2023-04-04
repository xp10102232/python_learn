import cv2
# 读取视频并显示
# 读取视频文件
# cap = cv2.VideoCapture('test.mp4')

# 读取摄像头
# cap = cv2.VideoCapture(0)

# # 读取视频流
video = "http://admin:admin@192.168.3.27:8081/"
cap = cv2.VideoCapture(video)

while True:
    success, img = cap.read()
    if success:
        cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release() 