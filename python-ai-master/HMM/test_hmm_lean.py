import numpy as np
from hmmlearn.hmm import MultinomialHMM, GMMHMM

if __name__ == "__main__":
    # # 离散 (discrete) HMM 
    # # 隐藏状态的数目
    # n_components = 3
    
    # # 状态的起始概率 pi
    # pi = np.ones(n_components)/n_components
    
    # # 状态转移矩阵
    # A = np.array([   
                # [0.4, 0.3, 0.3],
                # [0.3, 0.4, 0.3],
                # [0.3, 0.3, 0.4], 
                       # ])
                       
    # # 概率映射矩阵 B
    # M_O2S = np.zeros([3,8])
    # M_O2S[0,:6]=1/6.0
    # M_O2S[1,:4]=1/4.0
    # M_O2S[2,:8]=1/8.0
  
    # # 样本的种类 （这里有8种面）
    # n_features = 8
    
    # # 构建一个模型
    # m_discre_hmm = MultinomialHMM(n_components)
    # m_discre_hmm.startprob_ = pi
    # m_discre_hmm.transmat_ = A
    # m_discre_hmm.emissionprob_ = M_O2S
    # m_discre_hmm.n_features = 8
    
    # ''' 任务 0 生成一组数据 '''
    # datas,states=m_discre_hmm.sample(1000)
    # print(datas.shape)
    # print(states.shape)
    
    # datas = datas[:,0]
    # # 对状态变化进行统计
    # S =1
    # index_current_S = np.where(states==S)[0]
    # index_next_S = index_current_S+1
    # index_next_S= index_next_S[:-1]
    
    # temp = states[index_next_S]
    # print("State %d to"%(S))
    # print( "%.3f"%(np.where(temp==0)[0].shape[0]/np.shape(temp)[0]))
    # print( "%.3f"%(np.where(temp==1)[0].shape[0]/np.shape(temp)[0]))   
    # print( "%.3f"%(np.where(temp==2)[0].shape[0]/np.shape(temp)[0])) 
    
    # # 对不同状态下的数据分布进行统计
    # temp = datas[index_current_S]
    # print("datas at State %d "%(S))
    # print( "%.3f"%(np.where(temp==0)[0].shape[0]/np.shape(temp)[0]))
    # print( "%.3f"%(np.where(temp==1)[0].shape[0]/np.shape(temp)[0]))   
    # print( "%.3f"%(np.where(temp==2)[0].shape[0]/np.shape(temp)[0])) 
    # print( "%.3f"%(np.where(temp==3)[0].shape[0]/np.shape(temp)[0]))
    # print( "%.3f"%(np.where(temp==4)[0].shape[0]/np.shape(temp)[0]))   
    # print( "%.3f"%(np.where(temp==5)[0].shape[0]/np.shape(temp)[0])) 
    # print( "%.3f"%(np.where(temp==6)[0].shape[0]/np.shape(temp)[0]))
    # print( "%.3f"%(np.where(temp==7)[0].shape[0]/np.shape(temp)[0]))

    # ''' 任务1 计算一组数据的概率 '''
    # data = np.array([3,4,5])
    # data = np.expand_dims(data,axis=1)
    
    # prob = m_discre_hmm.score(data)
    # print("----log_prob----",prob)
    
    # ''' 任务 2  维特比译码'''
    # decode = m_discre_hmm.predict(data)
    # print("-----decode-------",decode)
    
    
    # '''任务 3 模型训练'''
    
    # # 构建另一个参数已知的HMM模型
    # # 状态的初始分布
    # pi2  =np.array([1.0/3.0,1.0/3.0,1.0/3.0])
    # # 各个状态之间的转移关系
    # # 行表示当前状态 列表示下一状态
    # A2 = np.array([   
                   # [0.8, 0.1, 0.1],
                   # [0.1, 0.8, 0.1],
                   # [0.1, 0.1, 0.8], 
                       # ])
    
    # # 观测样本与各个状态之间的概率映射关系矩阵                   
    # M_O2S_2 = np.zeros([3,8])
    # M_O2S_2[0,:6]=[0.8,0.04,0.04,0.04,0.04,0.04]
    # M_O2S_2[1,:4]=1/4.0
    # M_O2S_2[2,:8]=1/8.0
    
    # m_discre_hmm2 = MultinomialHMM(n_components)
    # m_discre_hmm2.startprob_ = pi2
    # m_discre_hmm2.transmat_ = A2
    # m_discre_hmm2.emissionprob_ = M_O2S_2
    # m_discre_hmm2.n_features = 8
    
    # # model_hmm1 生成一组训练、测试 数据
    # train_datas_hmm1 = []
    # for i in range(200):
        # data_hmm1,_ = m_discre_hmm.sample(100)
        # train_datas_hmm1.append(data_hmm1)
        
    # test_datas_hmm1 = []
    # for i in range(30):
        # data_hmm1,_ = m_discre_hmm.sample(100)
        # test_datas_hmm1.append(data_hmm1)
    
    # # model_hmm2 生成一组训练、测试 数据
    # train_datas_hmm2 = []
    # for i in range(200):
        # data_hmm2,_ = m_discre_hmm2.sample(100)
        # train_datas_hmm2.append(data_hmm2)
        
    # test_datas_hmm2 = []
    # for i in range(30):
        # data_hmm2,_ = m_discre_hmm2.sample(100)
        # test_datas_hmm2.append(data_hmm2)
        
   
    # # 训练模型 1  所有参数随机初始化
    # trained_model_1 = MultinomialHMM(n_components,n_iter=50, tol=0.001, verbose=True)
    # length_train_datas_hmm1 = []
    # for data in train_datas_hmm1:
        # length_train_datas_hmm1.append(np.shape(data)[0])
    
    # trained_model_1.fit(np.concatenate(train_datas_hmm1,axis=0),np.array(length_train_datas_hmm1))
    
   
    
    # # 训练模型2  所有参数随机初始化
    # trained_model_2 = MultinomialHMM(n_components,n_iter=50, tol=0.001, verbose=True)
    # length_train_datas_hmm2 = []
    # for data in train_datas_hmm2:
        # length_train_datas_hmm2.append(np.shape(data)[0])
    
    # trained_model_2.fit(np.concatenate(train_datas_hmm2,axis=0),np.array(length_train_datas_hmm2))
    
    
    # # 进行测试
    # N_test1 = len(test_datas_hmm1)
    # N_test2 = len(test_datas_hmm2)
    # labs = [0 for i in range(N_test1)] + [1 for i in range(N_test2)]
    # test_datas = test_datas_hmm1 + test_datas_hmm2
    
    # for i,test_data in  enumerate(test_datas):
        
        # score1 = trained_model_1.score(test_data)
        # score2 = trained_model_2.score(test_data)
        # det_lab = np.argmax([score1,score2])
        
        # print("%f  %f  det_lab=%d  true_lab = %d"%(score1,score2,det_lab,labs[i])) 
    
    
    # # 模型训练 指定初始化参数 以及只更新部分参数
    # # params='ste',  在训练过程中更新的参数 
    # # init_params='ste' 由模型自动初始化化的参数
    # trained_model_3 = MultinomialHMM(n_components,n_iter=50, tol=0.001, verbose=True,params='te',init_params='')
    
    # pi3  =np.array([0.2,0.7,0.1])
    # A3 = np.array([   
                   # [0.8, 0.2, 0],
                   # [0, 0.8, 0.2],
                   # [0, 0, 1], 
                       # ])
    
    # # 观测样本与各个状态之间的概率映射关系矩阵                   
    # M_O2S = np.zeros([3,8])
    # M_O2S[0,:6]=[0.8,0.04,0.04,0.04,0.04,0.04]
    # M_O2S[1,:4]=1/4.0
    # M_O2S[2,:8]=1/8.0
    
    # trained_model_3.startprob_=pi3
    # trained_model_3.transmat_ = A3
    # trained_model_3.emissionprob_ = M_O2S
    
    # trained_model_3.fit(np.concatenate(train_datas_hmm1,axis=0),np.array(length_train_datas_hmm1))
    
    # print(trained_model_3.startprob_)
    # print(trained_model_3.transmat_)
    # print(trained_model_3.emissionprob_)
    
    
    
    
    ''' GMM  HMM  测试'''
    n_components = 3
    n_mix = 3
    n_feature =2 
    m_GMMHMM1 = GMMHMM(n_components=3, n_mix=3,covariance_type='full')
    pi =  np.array([1.0/3.0,1.0/3.0,1.0/3.0])
    A = np.array([   
                   [0.8, 0.1, 0.1],
                   [0.8, 0.1, 0.1],
                   [0.8, 0.1, 0.1], 
                  ])
    hmm_mus = np.zeros([n_components,n_mix,n_feature])
    hmm_sigmas = np.zeros([n_components,n_mix,n_feature,n_feature])
    hmm_ws = np.zeros([n_components,n_mix])
    for i in range(n_components):
        # 3个高斯成分 K  特征维度是2  D
        mus = 0.6*np.random.random_sample([n_mix,n_feature])-0.3  #[K,D]
        sigmas= np.array([np.eye(n_feature,n_feature) for i in range(3)]) #[K,D,D]
        ws = np.array([0.3,0.5,0.2])                           #[K]  
        hmm_mus[i]=mus
        hmm_sigmas[i] = sigmas
        hmm_ws[i] = ws
    
    m_GMMHMM1.startprob_ = pi
    m_GMMHMM1.transmat_ = A
    m_GMMHMM1.weights_ = hmm_ws
    m_GMMHMM1.means_ = hmm_mus
    m_GMMHMM1.covars_ = hmm_sigmas
    m_GMMHMM1.n_features =2
    
    
    ''' 任务 0  生成数据'''
    datas,states = m_GMMHMM1.sample(10)
    print("-------datas------",datas.shape,datas)
    print("------ states-----",states) 

    ''' 任务1  计算前向概率'''
    obs = np.array([ [0.3,0.3], [0.1,0.1], [0.2,0.2]])
    prob_forward =m_GMMHMM1.score(obs)
    print("---log_prob-------",prob_forward)
    
    
    ''' 任务2 维特比译码 '''
    decode = m_GMMHMM1.predict(obs)
    print("-----decode-------",decode)
    
    
    ''' 任务3 模型训练 '''
    
    # 构建GMM-HMM2
    pi = np.array([0.6,0.2,0.2])
    
    A = np.array([   
                 [0.1, 0.1, 0.8],
                 [0.1, 0.8, 0.1],
                 [0.8, 0.1, 0.1], 
                 ])
   
    hmm_mus = np.zeros([n_components,n_mix,n_feature])
    hmm_sigmas = np.zeros([n_components,n_mix,n_feature,n_feature])
    hmm_ws = np.zeros([n_components,n_mix])
    for i in range(n_components):
        # 3个高斯成分 K  特征维度是2  D
        mus = 0.1*np.random.random_sample([n_mix,n_feature])-0.5  #[K,D]
        sigmas= np.array([np.eye(n_feature,n_feature) for i in range(3)]) #[K,D,D]
        ws = np.array([0.1,0.1,0.8])                          #[K]  
        hmm_mus[i]=mus
        hmm_sigmas[i] = sigmas
        hmm_ws[i] = ws
        
    m_GMMHMM2 = GMMHMM(n_components=3, n_mix=3,covariance_type='full')
    m_GMMHMM2.startprob_ = pi
    m_GMMHMM2.transmat_ = A
    m_GMMHMM2.weights_ = hmm_ws
    m_GMMHMM2.means_ = hmm_mus
    m_GMMHMM2.covars_ = hmm_sigmas
    m_GMMHMM2.n_features =2

    # 利用 GMMHMM1 与 GMMHMM2 分别生成两组训练和测试数据
    
    # model_hmm1 生成一组训练、测试 数据
    train_datas_hmm1 = []
    for i in range(100):
        data_hmm1,_ = m_GMMHMM1.sample(100)
        train_datas_hmm1.append(data_hmm1)
        
    test_datas_hmm1 = []
    for i in range(30):
        data_hmm1,_ = m_GMMHMM1.sample(100)
        test_datas_hmm1.append(data_hmm1)
    
    # model_hmm2 生成一组训练、测试 数据
    train_datas_hmm2 = []
    for i in range(100):
        data_hmm2,_ = m_GMMHMM2.sample(100)
        train_datas_hmm2.append(data_hmm2)
        
    test_datas_hmm2 = []
    for i in range(30):
        data_hmm2,_ = m_GMMHMM2.sample(100)
        test_datas_hmm2.append(data_hmm2)


    # 分别进行模型训练
    
    # 训练模型1 参数全部随机初始化
    train_GMMHMM1 = GMMHMM(n_components=3, n_mix=3,covariance_type='full',n_iter=50, tol=0.0001, verbose=True)
    length_train_datas_hmm1 = []
    for data in train_datas_hmm1:
        length_train_datas_hmm1.append(np.shape(data)[0])
    
    print("-------------Train model 1------------------")
    train_GMMHMM1.fit(np.concatenate(train_datas_hmm1,axis=0),np.array(length_train_datas_hmm1))
    
    # 训练模型2 参数全部随机初始化
    train_GMMHMM2 = GMMHMM(n_components=3, n_mix=3,covariance_type='full',n_iter=50, tol=0.0001, verbose=True)
    length_train_datas_hmm2 = []
    for data in train_datas_hmm2:
        length_train_datas_hmm2.append(np.shape(data)[0])
    
    print("-------------Train model 2------------------")
    train_GMMHMM2.fit(np.concatenate(train_datas_hmm2,axis=0),np.array(length_train_datas_hmm2))
    
    # 进行测试
    N_test1 = len(test_datas_hmm1)
    N_test2 = len(test_datas_hmm2)
    labs = [0 for i in range(N_test1)] + [1 for i in range(N_test2)]
    test_datas = test_datas_hmm1 + test_datas_hmm2
    
    for i,test_data in  enumerate(test_datas):
        
        score1 = train_GMMHMM1.score(test_data)
        score2 = train_GMMHMM2.score(test_data)
        det_lab = np.argmax([score1,score2])
        
        print("%f  %f  det_lab=%d  true_lab = %d"%(score1,score2,det_lab,labs[i])) 
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
                       
    