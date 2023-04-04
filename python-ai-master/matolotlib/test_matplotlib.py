import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import cv2 
if __name__ == "__main__":

    # 最基本的使用
    data = np.array([1,2,3,4,5,6,7])
    plt.figure(1)
    plt.plot(data)
    plt.show()
    #######默认##############
    # fig = plt.figure(1)
    # ax = fig.add_subplot(1, 1,1)
    # ax.plot(data)
    
    plt.savefig("1.jpg")

    # 分屏显示
    fig = plt.figure(2)
    ax1 = fig.add_subplot(2, 2,1)
    ax1.plot(np.arange(10))
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(np.arange(60))
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(np.arange(100))
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(np.arange(40))
    plt.savefig("2.jpg")
    
  
    # 分屏2 不规则布局
    fig = plt.figure(3)
    ax1 = plt.subplot2grid((3,3),(0,0),colspan=3)
    ax2 = plt.subplot2grid((3,3),(1,0),colspan=2,rowspan=2)
    ax3 = plt.subplot2grid((3,3),(1,2),rowspan=2)
    
    ax1.plot(np.random.randn(50).cumsum(), 'k--')
    ax2.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)
    ax3.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
    plt.savefig("3.jpg")
   
    # plot 函数的样例
    data = np.random.randn(50).cumsum()
    fig = plt.figure(4)
    # 直接画
    ax1 = fig.add_subplot(4, 1,1)
    ax1.plot(data)
    
    # 添加横坐标
    ax2 = fig.add_subplot(4, 1,2)
    x = np.linspace(0,1,50)
    ax2.plot(x,data)

    # 添加横坐标+线型（格式 [color][marker][line]）
    ax3 = fig.add_subplot(4, 1,3)
    ax3.plot(x,data,'rs--')

    # 添加横坐标+线型（格式 [color][marker][line]）
    # + 更为详细的参数
    ax4 = fig.add_subplot(4, 1,4)
    ax4.plot(x,data,color=(0,0.5,0),marker ='*',linestyle = '-',linewidth=3, markersize=10)
    
    plt.savefig("4.jpg")
    # 直方图示例
    data = np.random.randn(100)
    
    fig = plt.figure(5)

    # 整数bin
    ax1 = fig.add_subplot(2,1,1)
    ax1.hist(data, bins=10, color='k', alpha=0.3)
    
    # 区间bin
    ax2 = fig.add_subplot(2,1,2)
    m_bin = np.linspace(-3,3,10)
    ax2.hist(data, bins=m_bin,ec="yellow", fc="k",alpha=0.9)


    # 散点图示例
    data = np.random.randn(2,50)
    x = data[0,:]
    y = data[1,:]
    value = np.random.rand(50)
    fig = plt.figure(6)
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)

    # 简单显示 
    ax1.scatter(x,y,s=56,c='r',marker='o')

    # 颜色以及点的尺寸会随value的变化而变化
    sizes = ((value)*16)**2
    ax2.scatter(x,y,s=sizes,c=value,marker='o',cmap='viridis',alpha=0.3)

    # 显示两组数据
    ax3.scatter(x,y,s=56,c='r',marker='o')
    data = np.random.randn(2,50)
    x = data[0,:]
    y = data[1,:]
    ax3.scatter(x,y,s=66,c='g',marker='*')

    # 图例的示例
    fig = plt.figure(7)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    # 多plot
    data1 = np.random.randn(50).cumsum()
    data2 = np.random.randn(50).cumsum()
    ax1.plot(data1,'rs--',label = 'data1')
    ax1.plot(data2,'go-',label = 'data2')
    ax1.legend(loc='best')

    # 多scatter
    data = np.random.randn(2,50)
    ax2.scatter(data[0,:],data[1,:],s=40,c='g',marker='*',label = 'data1')
    data = np.random.randn(2,50)
    ax2.scatter(data[0,:],data[1,:],s=60,c='b',marker='o',label = 'data2')
    ax2.legend(loc='best')


    # 坐标轴及标题设置
    # 字体
    zhfont1 = matplotlib.font_manager.FontProperties(fname="NotoSansCJK-Bold.ttc", size=16)
    fig = plt.figure(8)
    ax1 = fig.add_subplot(1,1,1)
    
    # 多plot
    data1 = np.random.randn(50).cumsum()
    data2 = np.random.randn(50).cumsum()
    ax1.plot(data1,'rs--',label = 'data1')
    ax1.plot(data2,'go-',label = 'data2')
    ax1.legend(loc='best')

    ax1.set_xlim([0,49])
    ax1.set_ylim([-10,10])
    ax1.set_xlabel('x轴',fontproperties=zhfont1)
    ax1.set_ylabel('y轴',fontproperties=zhfont1)
    ax1.set_title('多plot示例',fontproperties=zhfont1)



    fig = plt.figure(9)
    # 多scatter
    data = np.random.randn(2,50)
    plt.scatter(data[0,:],data[1,:],s=40,c='g',marker='*',label = 'data1')
    data = np.random.randn(2,50)
    plt.scatter(data[0,:],data[1,:],s=60,c='b',marker='o',label = 'data2')
    plt.legend(loc='best')
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.xlabel('x轴',fontproperties=zhfont1)
    plt.ylabel('y轴',fontproperties=zhfont1)
    plt.title("多散点描述",fontproperties=zhfont1)
    plt.savefig("9.jpg")
   

    # 图像显示
    fig = plt.figure('IMG')
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)

    img = cv2.imread("0.jpg")
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 显示彩色图
    ax1.imshow(img1)
    
    # 显示灰度图
    im_g = ax2.imshow(gray,cmap='gray',vmin=0,vmax =255)
    plt.colorbar(im_g,ax =ax2 )
    
    # 将一个随机矩阵 进行颜色显示
    data = np.log(np.random.rand(256,256))
    img_r = ax3.imshow(data)
    plt.colorbar(img_r,ax = ax3)

    plt.savefig("img.jpg")


    plt.show()
    