import numpy as np
from NN_BP import *

def load_mnist(file_data,file_lab):
    # 加载训练数据
    data = np.load(file_data)
    lab = np.load(file_lab)
    N,D = np.shape(data)
    
    # 构造 one-hot 标签
    lab_onehot = np.zeros([N,10])
    for i in range(N):
        id = int(lab[i,0])
        lab_onehot[i,id]=1
    data = (data.astype(np.float)/255.0)
    return data,lab_onehot
    

if __name__=="__main__":
    
    # 加载训练数据
    train_data,train_lab_onehot=load_mnist("train_data.npy","train_lab.npy")
    N,D = np.shape(train_data) 
    
    # 搭建网络
    # 定义网络结构
    list_num_hidden=[30,5,10]
    
    # list_act_funs =[sigmod,sigmod,no_active]
    # list_de_act_funs=[de_sigmoid,de_sigmoid,de_no_active]
    
    # # 定义损失函数
    # loss_fun = loss_L2
    # de_loss_fun=de_loss_L2
    
    list_act_funs =[relu,relu,no_active]
    list_de_act_funs=[de_relu,de_relu,de_no_active]
    # 定义损失函数
    loss_fun = loss_CE
    de_loss_fun=de_loss_CE
    
    layers = bulid_net(D,list_num_hidden,
          list_act_funs,list_de_act_funs)
          
    # 进行训练
    n_epoch = 50
    batchsize =20    
    N_batch = N//batchsize
    for i in range(n_epoch):
        # 数据打乱
        rand_index  = np.random.permutation(N).tolist()
        # 每个batch 更新一下weight
        loss_sum =0
        for j in range(N_batch):
            index = rand_index[j*batchsize:(j+1)*batchsize]
            batch_datas = train_data[index]
            batch_labs = train_lab_onehot[index]
            layers,loss = updata_wb(batch_datas,batch_labs,layers,loss_fun,de_loss_fun,alpha=0.001)
            # print("epoch %d  batch %d  loss %.2f"%(i,j,loss/batchsize))
            loss_sum = loss_sum+loss
            
        error = test_accuracy(train_data,train_lab_onehot,layers)
        print("epoch %d  error  %.2f%%  loss_all %.2f"%(i,error*100,loss_sum/(N_batch*batchsize)))
        
    np.save("model.npy",layers)    
    
    # 加载测试数据
    test_data,test_lab_onehot=load_mnist("test_data.npy","test_lab.npy")
    layers = np.load("model.npy",allow_pickle=True)
   
    error = test_accuracy(test_data,test_lab_onehot,layers)
    print("Accuarcy on Test Data %.2f %%"%((1-error)*100))
    
    
    
    
    