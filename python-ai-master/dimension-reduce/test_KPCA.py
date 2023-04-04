import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x1, x2,a = 0.25,r=3):
    x = np.dot(x1, x2)
    return np.tanh(a*x+r)

def linear(x1,x2,a=1,c=0,d=1):
    x = np.dot(x1, x2)
    x = np.power((a*x+c),d)
    return x

def rbf(x1,x2,gamma = 3):
    x = np.dot((x1-x2),(x1-x2))
    x = np.exp(-gamma*x)
    return x
    
    
def kpca(data, n_dims=2, kernel = rbf):
    
    N,D = np.shape(data)
    K = np.zeros([N,N])
    
    # 利用核函数计算K
    for i in range(N):
        for j in range(N):
            K[i,j]=kernel(data[i],data[j])
    
    # 对K进行中心化
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    #计算特征值和特征向量
    eig_values, eig_vector = np.linalg.eig(K)
    idx = np.argsort(-eig_values)[:n_dims] # 从大到小排序
   
    # 选取较大的特征值
    eigval = eig_values[idx]
    eigvector = eig_vector[:, idx]  #[N,d]
    
    # 进行正则
    eigval = eigval**(1/2)
    u = eigvector/eigval.reshape(-1,n_dims) # u [N,d]
    
    # 进行降维
    data_n = np.dot(K, u)
    return data_n




def draw_pic(datas,labs):
    plt.cla()
    unque_labs = np.unique(labs)
    colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1,len(unque_labs))]
    p=[]
    legends = []
    for i in range(len(unque_labs)):
        index = np.where(labs==unque_labs[i])
        pi = plt.scatter(datas[index, 0], datas[index, 1], c =[colors[i]] )
        p.append(pi)
        legends.append(unque_labs[i])
    
    plt.legend(p, legends)
    plt.show()
    
    
    
    


if __name__ == "__main__":
    
    # # 加载数据
    # data = np.loadtxt("iris.data",dtype="str",delimiter=',')
    # feas = data[:,:-1]
    # feas = np.float32(feas)
    # labs = data[:,-1]
    # kpca(feas)
  
    # # 进行降维
    # data_2d = kpca(feas, n_dims=2,kernel=rbf)
    
    # #绘图
    # draw_pic(data_2d,labs)
    
    
    
    a1 = np.random.rand(100, 2)
    a1 =a1+np.array([4,0])
    
    a2 = np.random.rand(100, 2)
    a2 =a2+np.array([-4,0])
    
    a = np.concatenate((a1,a2),axis=0)
    
    b = np.random.rand(200, 2)
    
    data = np.concatenate((a,b),axis=0)
    labs =np.concatenate((np.zeros(200),np.ones(200)),axis=0)
    draw_pic(data,labs)
    
    
    data_2d = kpca(data, n_dims=2,kernel=rbf)
    
    draw_pic(data_2d,labs)
    
    
    
   
    
    
    
    
    
    
    
    