import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


def data_generate():
    N1 = 100
    center_1 = []
    for i in range(N1):
        # 随机生成角度
        th = random.uniform(0,2*3.14)
        r = random.uniform(2.3,2.4)
        x_ = r*np.cos(th)
        y_ = r*np.sin(th)
        center_1.append((x_,y_))

    N2 = 20
    center_2 = []
    for i in range(N2):
        # 随机生成角度
        th = random.uniform(0,2*3.14)
        r = random.uniform(0,0.25)
        x_ = -1+r*np.cos(th)
        y_ = 1+r*np.sin(th)
        center_2.append((x_,y_))
    
    
    N3 = 20
    center_3 = []
    for i in range(N3):
        # 随机生成角度
        th = random.uniform(0,2*3.14)
        r = random.uniform(0,0.25)
        x_ = 1+r*np.cos(th)
        y_ = 1+r*np.sin(th)
        center_3.append((x_,y_))
    

    N4 = 50
    center_4 = []
    for i in range(N4):
        # 随机生成角度
        th = random.uniform(3.14*240/180,3.14*300/180)
        r = random.uniform(1.3,1.4)
        x_ = r*np.cos(th)
        y_ = r*np.sin(th)
        center_4.append((x_,y_))
    
    center5 = center_1+center_2+center_3+center_4

    return np.array(center5)

def draw(datas,clusters,str_title=""):
    
    N_cluster = len(clusters)
    plt.cla()
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1,N_cluster)]
    
    for i,cluster in enumerate(clusters):
        datas_draw = datas[cluster,:]
        datas_draw = datas_draw[:,:2]
        plt.scatter(datas_draw[:,0],datas_draw[:,1],s=20,color=colors[i])

    plt.title(str_title)
   
    plt.show()



# 对一个簇 进行分裂
# cluster 一个簇中的样本的编号
# dis_matrix 全部样本 两两之间的距离
def split_one_cluster(cluster,dis_matrix):
    # 在一个簇中找到一个离群点,这个点距离其他点的平均距离最远
    temp_dis_matrix = dis_matrix[cluster][:,cluster]
    max_dis_index = np.argmax(np.mean(temp_dis_matrix,axis=1))
    id_split = cluster[max_dis_index]

    # 将簇先分成两个簇，离群点一个簇，其他点一个簇
    split_cluster = [id_split]
    last_cluster = cluster.copy()
    last_cluster.pop(max_dis_index)

    while True:
        flag_split = False
        for i in range(len(last_cluster))[::-1]:
            # 遍历其他点簇 中的所有点
            p = last_cluster[i]
            # 计算点p 和 split 中点的距离 
            dis_p_split = dis_matrix[p,split_cluster]
            # 计算点 p 和 last 中 其他点的距离
            point_left = last_cluster.copy()
            point_left.pop(i)
            dis_p_last = dis_matrix[p,point_left]

            # 如果点p 距离 split 更近
            if np.mean(dis_p_split) <= np.mean(dis_p_last):
                # 那么把点p 加入到 split 簇中
                split_cluster.append(p)
                last_cluster.pop(i)
                flag_split = True
                break
        
        # 如果遍历一轮没有找到新的分离点，则分裂结束
        if flag_split == False:
            break
    return split_cluster,last_cluster

# 从一组簇中 找到分离度最大的一个簇 进行分裂       
def get_max_separation_cluster(clusters,dis_matrix):
    dgree_separation = []
    for cluster in clusters:
        temp_dis_matrix = dis_matrix[cluster][:,cluster]
        dgree_separation.append(np.max(temp_dis_matrix))
    return np.argmax(dgree_separation)



if __name__ == "__main__":
  


    # 读取数据
    a = np.random.multivariate_normal([3,3], [[.5,0],[0,.5]], 100)
    b = np.random.multivariate_normal([0,0], [[0.5,0],[0,0.5]], 100)
    c = np.random.multivariate_normal([3,0], [[0.5,0],[0,0.5]], 100)
    d = np.random.multivariate_normal([0,3], [[0.5,0],[0,0.5]], 100)
    data = np.r_[a,b,c,d]


    # data= data_generate()
    print(data.shape)
    N,D= np.shape(data)
    
    # 计算数据点两两之间的距离
    tile_x = np.tile(np.expand_dims(data,1),[1,N,1]) # N, N,D
    tile_y = np.tile(np.expand_dims(data,0),[N,1,1]) # N, N,D
    dis_matrix = np.linalg.norm((tile_x-tile_y),axis=-1)

    # 从初始化时所有的样本点在一个类中
    clusters = [[i for i in range(N)]]
    K = 4
    draw(data,clusters,str_title="")
    while True:
        # 找到区分度最大的一个簇
        index_sel = get_max_separation_cluster(clusters,dis_matrix)
        #对这个簇进行分裂
        c_1,c_2 = split_one_cluster(clusters[index_sel],dis_matrix)

        # 删除分裂前的一个簇 添加分裂后的两个簇
        clusters.pop(index_sel)
        clusters.append(c_1)
        clusters.append(c_2)
        
        # 显示结果
        draw(data,clusters,str_title="")
        plt.show()
        if len(clusters)>=K:
            break
    
    # 显示聚类结果
    cluster_labels = np.zeros(N)
    for i in range(len(clusters)):
        cluster_labels[clusters[i]] = i
    print(cluster_labels)

    draw(data,clusters,str_title="")
    plt.show()









  







