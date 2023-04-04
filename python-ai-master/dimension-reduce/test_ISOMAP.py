import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve
from sklearn.manifold import Isomap
from tqdm import tqdm

# x 维度 [N,D]
def cal_pairwise_dist(x):
    
    N,D = np.shape(x)
    
    dist = np.zeros([N,N])
    
    for i in range(N):
        for j in range(N):
            # dist[i,j] = np.dot((x[i]-x[j]),(x[i]-x[j]).T)
            dist[i,j] = np.sqrt(np.dot((x[i]-x[j]),(x[i]-x[j]).T))
            # dist[i,j] = np.sum(np.abs(x[i]-x[j]))
    
    #返回任意两个点之间距离
    return dist
    
    
    
# 构建最短路径图
def floyd(D,n_neighbors=15):
    Max = np.max(D)*1000
    n1,n2 = D.shape
    k = n_neighbors
    D1 = np.ones((n1,n1))*Max
    D_arg = np.argsort(D,axis=1)
    for i in range(n1):
        D1[i,D_arg[i,0:k+1]] = D[i,D_arg[i,0:k+1]]
    for k in tqdm(range(n1)):
        
        for i in range(n1):
            for j in range(n1):
                if D1[i,k]+D1[k,j]<D1[i,j]:
                    D1[i,j] = D1[i,k]+D1[k,j]
    return D1


def my_mds(dist, n_dims):
    # dist (n_samples, n_samples)
    dist = dist**2
    n = dist.shape[0]
    T1 = np.ones((n,n))*np.sum(dist)/n**2
    T2 = np.sum(dist, axis = 1)/n
    T3 = np.sum(dist, axis = 0)/n

    B = -(T1 - T2 - T3 + dist)/2

    eig_val, eig_vector = np.linalg.eig(B)
    index_ = np.argsort(-eig_val)[:n_dims]
    picked_eig_val = eig_val[index_].real
    picked_eig_vector = eig_vector[:, index_]

    return picked_eig_vector*picked_eig_val**(0.5)

    
 
# dist N*N 距离矩阵样本点两两之间的距离 
# n_dims 降维
# 返回 降维后的数据
# def my_mds(dist, n_dims):
    
    # n,n = np.shape(dist)
    
    # dist[dist < 0 ] = 0
    
    # T1 = np.ones((n,n))*np.sum(dist)/n**2
    # T2 = np.sum(dist, axis = 1, keepdims=True)/n
    # T3 = np.sum(dist, axis = 0, keepdims=True)/n

    # B = -(T1 - T2 - T3 + dist)/2

    # eig_val, eig_vector = np.linalg.eig(B)
    # index_ = np.argsort(-eig_val)[:n_dims]
    # picked_eig_val = eig_val[index_].real
    # picked_eig_vector = eig_vector[:, index_]
    
    # return picked_eig_vector*picked_eig_val**(0.5)

def my_Isomap(D,n=2,n_neighbors=30):

    D_floyd=floyd(D, n_neighbors)
    data_n = my_mds(D_floyd, n_dims=n)
    return data_n



def scatter_3d(X, y):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.hot)
    ax.view_init(10, -70)
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    plt.show(block=False)










if __name__ == "__main__":
    
    
    X, Y = make_s_curve(n_samples = 500,
                           noise = 0.1,
                           random_state = 42)
    scatter_3d(X,Y)
    
    # 计算距离
    dist = cal_pairwise_dist(X)
    
    
    
    # MDS 降维
    data_MDS = my_mds(dist, 2)
    
    plt.figure()
    plt.title("my_MSD")
    plt.scatter(data_MDS[:, 0], data_MDS[:, 1], c = Y)
    plt.show(block=False)
    
    
    # ISOMAP 降维
    data_ISOMAP = my_Isomap(dist, 2, 10)
   
    plt.figure()
    plt.title("my_Isomap")
    plt.scatter(data_ISOMAP[:, 0], data_ISOMAP[:, 1], c = Y)
    plt.show(block=False)
    
    data_ISOMAP2 = Isomap(n_neighbors = 10, n_components = 2).fit_transform(X)
    
    plt.figure()
    plt.title("sk_Isomap")
    plt.scatter(data_ISOMAP2[:, 0], data_ISOMAP2[:, 1], c = Y)
    plt.show(block=False)
    
    plt.show()
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    # # 加载数据
    # data = np.loadtxt("iris.data",dtype="str",delimiter=',')
    # feas = data[:,:-1]
    # feas = np.float32(feas)
    # labs = data[:,-1]
    
    # # 计算距离
    # dist = cal_pairwise_dist(feas)
    
    # # 进行降维
    # data_2d = my_mds(dist, 2)
    
    # #绘图
    # draw_pic(data_2d,labs)
    
    
    