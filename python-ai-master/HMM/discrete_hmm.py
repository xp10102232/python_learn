import numpy as np

def prob_O2S(model,o):
    M_O2S = model["M_O2S"]
    return M_O2S[:,int(o)]


def gen_one_sample_from_Prob_list(Prob_list):
    N_segment = np.shape(Prob_list)[0]
    
    # 将[0,1]的区间分为 N_segment 段
    prob_segment = np.zeros(N_segment)
    # 例如 Prob_list = [  0.3,0.3,0.4]
    #      prob_segment_segment = [0.3,0,6,1]
    for i in range(N_segment):
        prob_segment[i] = prob_segment[i-1]+ Prob_list[i]
    
    S =0
    # 生成0,1之间的随机数
    data = np.random.rand()
    # 查看生成的数值位于哪个段中
    for i in range(N_segment):
        if data <= prob_segment[i]:
            S = i
            break
    return S
    
def gen_samples_from_HMM(model,N):
    M_O2S = model["M_O2S"]
   
    datas = np.zeros(N)
    stats = np.zeros(N)
    
    # 得到初始状态，并根据初始状态生成一个样本
    init_S = gen_one_sample_from_Prob_list(model["pi"])
    stats[0] = init_S
    
    # 根据初始状态，生成一个数据
    datas[0] = gen_one_sample_from_Prob_list(M_O2S[int(stats[0])]) 
    
    #生成其他样本 
    for i in range(1,N):
        # 根据前一状态，生成当前状态
        stats[i] = gen_one_sample_from_Prob_list(model["A"][int(stats[i-1])])
        # 根据当前状态生成一个数据
        datas[i] = gen_one_sample_from_Prob_list(M_O2S[int(stats[i])])
    return datas,stats


# 前向后向算法

# 前向算法  alpha = [N_sample,N_stats]
# 表示已知前t个样本的的情况下，第t个样本
# 属于状态i的概率
def calc_alpha(model,observations):
    o = observations
    N_samples = np.shape(o)[0]
    N_stats = np.shape(model["pi"])[0]
    
    # alpha 初始化
    alpha = np.zeros([N_samples,N_stats])
    
    # 计算第0个样本属于第i个状态的概率
    alpha[0] = model["pi"]*model["B"](model,o[0])
   
    # 计算其他时刻的样本属第i个状态的概率
    for t in range(1,N_samples):
        s_current = np.dot(alpha[t-1],model["A"])
        alpha[t] = s_current*model["B"](model,o[t])
   
    return alpha
    
# 后向算法
def calc_beta(model,observations):
    o = observations
    N_samples = np.shape(o)[0]
    N_stats = np.shape(model["pi"])[0]
    
    beta = np.zeros([N_samples,N_stats])
    
    # 反向初始值
    beta[-1] = 1
    
    for t in range(N_samples-2,-1,-1):
        # 由t+1时刻的beta以及t+1时刻的观测值计算
        # t+1时刻的状态值
        s_next = beta[t+1]*model["B"](model,o[t+1])
        beta[t] = np.dot(s_next,model["A"].T)
    return beta    
        

def forward(model,observations):
    o = observations
    
    # 计算前向概率
    alpha =  calc_alpha(model,o)
    prob_seq_f = np.sum(alpha[-1])

    return np.log(prob_seq_f)
    
def backward(model,observations):
    o = observations
    
    # 计算后向概率
    beta =  calc_beta(model,o)
    s_next = beta[0]*model["B"](model,o[0]) 
    prob_seq_b = np.dot(s_next,model["pi"])
    
    return np.log(prob_seq_b)


# 维特比译码    
def decoder(model,observations):
    o = observations
    N_samples = np.shape(o)[0]
    N_stats = np.shape(model["pi"])[0]
    
    # 记录了从t-1 到 t时刻，状态i
    # 最可能从哪个状态（假设为j）转移来的
    psi = np.zeros([N_samples,N_stats])
    
    # 从t-1 到 t 时刻状态 状态j到状态i的最大的转移概率
    delta = np.zeros([N_samples,N_stats])
    
    # 初始化
    delta[0] = model["pi"]*model["B"](model,o[0])
    psi[0]=0
    
    # 递推填充 delta 与 psi
    for t in range(1,N_samples):
        for i in range(N_stats):
            states_prev2current = delta[t-1] * model["A"][:,i]
            delta[t][i] = np.max(states_prev2current)
            psi[t][i] = np.argmax(states_prev2current)
        
        delta[t] = delta[t]*model["B"](model,o[t])
            
    # 反向回溯寻找最佳路径
    path = np.zeros(N_samples)
    path[-1] = np.argmax(delta[-1])
    prob_max = np.max(delta[-1])
    
    for t in range(N_samples-2,-1,-1):
        path[t] = psi[t+1][int(path[t+1])]
    
    return prob_max, path 
        

def calcxi(model,observations,alpha,beta):

    o=observations
    N_samples = np.shape(o)[0]
    N_stats = np.shape(model["pi"])[0]

    xi = np.zeros([N_samples-1,N_stats,N_stats])

    for t in range(N_samples-1):
        temp = np.zeros([N_stats,N_stats])
        
        t_alpha = np.tile( np.expand_dims(alpha[t,:],axis=1),(1,N_stats))
        t_beta = np.tile(beta[t+1,:],(N_stats,1))
        t_b = np.tile(model["B"](model,o[t+1]),(N_stats,1))
        
        temp = t_alpha*model["A"]*t_beta*t_b
        
        temp = temp/np.sum(temp)
 
        xi[t]=temp
    return xi


def calcxi_2(model,observations,alpha,beta):
    
    o=observations
    N_samples = np.shape(o)[0]
    N_stats = np.shape(model["pi"])[0]
    xi = np.zeros((N_samples,N_stats,N_stats))
    for t in range(N_samples-1):
        denom = 0.0
        
        for i in range(N_stats):
            for j in range(N_stats):
                thing = 1.0
                thing *= alpha[t][i]
                thing *= model["A"][i][j]
                thing *= model["B"](model,o[t+1])[j]
                thing *= beta[t+1][j]
                denom += thing
                
        for i in range(N_stats):
            for j in range(N_stats):
                numer = 1.0
                numer *= alpha[t][i]
                numer *= model["A"][i][j]
                numer *= model["B"](model,o[t+1])[j]
                numer *= beta[t+1][j]
                xi[t][i][j] = numer/denom

    return xi


def calcgamma(alpha,beta):
    gamma = alpha*beta

    gamma = gamma/np.sum(gamma,axis=1,keepdims=True)

    return gamma    
    

def update_pi(collect_gamma):
    
    N_datas = len(collect_gamma)
    _,N_stats = np.shape(collect_gamma[0])
    
    sum_gamma_1 = np.zeros(N_stats)
    
    for gamma in collect_gamma:
        sum_gamma_1 = sum_gamma_1+gamma[0]
    
    pi = sum_gamma_1/N_datas
    return pi
        
def update_A(collect_gamma,collect_xi):
    
    _,N_stats = np.shape(collect_gamma[0])
    
    sum_xi = np.zeros([N_stats,N_stats])
    
    sum_gamma = np.zeros(N_stats)
    
    for xi in collect_xi:
        sum_xi = sum_xi + np.sum(xi,axis=0)
    
    # sum_gamma = np.sum(sum_xi,axis=1,keepdims=True)
    for gamma in collect_gamma:
        sum_gamma = sum_gamma+ np.sum(gamma[:-1],axis=0)

    sum_gamma = np.tile(np.expand_dims(sum_gamma,axis=1),(1,N_stats))

    A = sum_xi/sum_gamma
    return A
    
  
    
    
def update_M_O2S(datas,model,collect_gamma):
    
    N_datas = len(datas)
    
    N_state,N_symbol = np.shape(model["M_O2S"])
    
    sum_gamma = np.zeros(N_state)
    for gamma in collect_gamma:
        sum_gamma = sum_gamma+ np.sum(gamma,axis=0)
    
    sum_M_O2S = np.zeros([N_state,N_symbol])
   
    # 遍历每一种观测数据
    for k in range(N_symbol):
        
        for d in range(N_datas):
            o = datas[d]
            gamma = collect_gamma[d]
            
            index_k = np.where(o==k)[0]
            # 计算每种观测样本 属于每个状态的概率
            sum_M_O2S[:,k] = sum_M_O2S[:,k] + np.sum(gamma[index_k],axis=0)
            
    sum_gamma = np.tile(np.expand_dims(sum_gamma,axis=1),(1,N_symbol))
    
    new_M_O2S = sum_M_O2S/sum_gamma
    return new_M_O2S
            
            
        
        
        
        
    
    
    
    




# 训练数据是一个numpy的列表
def train_step(train_datas,model):
    
    # 对每一条数据进行处理
   
    collect_xi = []
    collect_gamma = []
    for datas in train_datas:
        # 计算前向概率
        alpha = calc_alpha(model,datas)
        
        # 计算后向概率
        beta = calc_beta(model,datas)
        
        # 计算xi
        xi = calcxi(model,datas,alpha,beta)
        
        # 计算gamma
        gamma = calcgamma(alpha,beta)
        
        # 对处理的结果进行拼接
        collect_gamma.append(gamma)
        collect_xi.append(xi)
    
    # 计算新的参数
    new_A = update_A(collect_gamma,collect_xi)
    new_pi = update_pi(collect_gamma)
    new_M_O2S = update_M_O2S(train_datas,model,collect_gamma)

    return new_A,new_pi,new_M_O2S
    

def compute_prob_for_datas(model,datas):
    results = 0
    
    for o in datas:
        results += forward(model,o)
        
    return results
    
    
    

def train(train_datas,model,n_iteration):
    
    new_model = model.copy()
    
    prob_old = compute_prob_for_datas(model,train_datas)
    print("Prob_first",prob_old)
    
    
    for i in range(n_iteration):
       
        new_A,new_pi,new_M_O2S = train_step(train_datas,new_model)
        
        new_model["A"] =  new_A
        new_model["pi"] = new_pi
        new_model["M_O2S"]= new_M_O2S
        new_model["B"] = prob_O2S
        
        prob_new =  compute_prob_for_datas(new_model,train_datas)
        print("it %d prob %f"%(i,prob_new))
       

        if prob_new>prob_old:
            prob_old = prob_new
            model = new_model
        else: 
            break
            
        
        
    return model
        
        


if __name__ == "__main__":
    
    ''' 设计一个 HMM 模型'''
    # 构建一个参数已知的HMM模型
    model_hmm1 = dict()
    # 状态的初始分布
    model_hmm1["pi"] =np.array([1.0/3.0,1.0/3.0,1.0/3.0])
    # 各个状态之间的转移关系
    # 行表示当前状态 列表示下一状态
    model_hmm1["A"] = np.array([   
                        [0.4, 0.3, 0.3],
                        [0.3, 0.4, 0.3],
                        [0.3, 0.3, 0.4], 
                       ])
    
    # 观测样本与各个状态之间的概率映射关系矩阵                   
    M_O2S = np.zeros([3,8])
    M_O2S[0,:6]=1/6.0
    M_O2S[1,:4]=1/4.0
    M_O2S[2,:8]=1/8.0
    model_hmm1["M_O2S"] = M_O2S
    
    # 计算观测样本O属于状态S的概率的函数
    model_hmm1["B"] = prob_O2S
    
    ''' 任务1 生成一条满足参数已知HMM分布的数据 '''
    # datas,states = gen_samples_from_HMM(model_hmm1,2000)
    # print(datas)
    # print(states)
    
    
    # # 对状态变化进行统计
    # S =2
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
    
    
    ''' 任务2 利用前向 后向算法 计算一个生成一个序列的概率'''
    # datas = np.array([1,3,4])
    # print("alpha")
    # print(calc_alpha(model_hmm1,datas))
    
    # print("beta")
    # print(calc_beta(model_hmm1,datas))
    
    # print("---------------")
    # print("forward",forward(model_hmm1,datas))
    # print("backward",backward(model_hmm1,datas))
    
    
    # 构建另一个参数已知的HMM模型
    model_hmm2 = dict()
    # 状态的初始分布
    model_hmm2["pi"] =np.array([1.0/3.0,1.0/3.0,1.0/3.0])
    # 各个状态之间的转移关系
    # 行表示当前状态 列表示下一状态
    model_hmm2["A"] = np.array([   
                        [0.8, 0.1, 0.1],
                        [0.1, 0.8, 0.1],
                        [0.1, 0.1, 0.8], 
                       ])
    
    # 观测样本与各个状态之间的概率映射关系矩阵                   
    M_O2S_2 = np.zeros([3,8])
    M_O2S_2[0,:6]=[0.8,0.04,0.04,0.04,0.04,0.04]
    M_O2S_2[1,:4]=1/4.0
    M_O2S_2[2,:8]=1/8.0
    model_hmm2["M_O2S"] = M_O2S_2
    model_hmm2["B"] = prob_O2S
    
    
    
    # 分别用2个HMM生成2个数据序列
    # datas_hmm1,_ = gen_samples_from_HMM(model_hmm1,100)
    # datas_hmm2,_ = gen_samples_from_HMM(model_hmm2,100)
    
    # p_d1m1 = forward(model_hmm1,datas_hmm1)
    # p_d1m2 = forward(model_hmm2,datas_hmm1)
    
    # p_d2m1 = forward(model_hmm1,datas_hmm2)
    # p_d2m2 = forward(model_hmm2,datas_hmm2)
    
    # print("p_d1m1",p_d1m1)
    # print("p_d1m2",p_d1m2)
    # print("----------------------------------")
    # print("p_d2m1",p_d2m1)
    # print("p_d2m2",p_d2m2)
    
    '''任务3 利用维特比算法使用'''
    # datas = np.array([0,1,0,1])
    # _,path = decoder(model_hmm1,datas)
    # print(path)
    
    
    ''' 任务4 训练 测试 '''
    
    # model_hmm1 生成一组训练、测试 数据
    train_datas_hmm1 = []
    for i in range(200):
        data_hmm1,_ = gen_samples_from_HMM(model_hmm1,100)
        train_datas_hmm1.append(data_hmm1)
        
    test_datas_hmm1 = []
    for i in range(30):
        data_hmm1,_ = gen_samples_from_HMM(model_hmm1,100)
        test_datas_hmm1.append(data_hmm1)
    
    # model_hmm2 生成一组训练、测试 数据
    train_datas_hmm2 = []
    for i in range(200):
        data_hmm2,_ = gen_samples_from_HMM(model_hmm2,100)
        train_datas_hmm2.append(data_hmm2)
        
    test_datas_hmm2 = []
    for i in range(30):
        data_hmm2,_ = gen_samples_from_HMM(model_hmm2,100)
        test_datas_hmm2.append(data_hmm2)


    # 初始化一个 hmm 
    init_hmm = dict()
    # 初始化 pi
    pi = np.random.random_sample((3))
    pi = pi / sum(pi)
    init_hmm["pi"] = pi
    
    # 初始化 A
    a = np.random.random_sample((3, 3))
    row_sums = a.sum(axis=1)
    a = a / row_sums[:, np.newaxis]  
    init_hmm["A"] = a
    
    # 初始化 B
    b = np.random.random_sample((3, 8))
    row_sums = b.sum(axis=1)
    b = b / row_sums[:, np.newaxis]
    init_hmm["M_O2S"] = b
    init_hmm["B"] = prob_O2S

    # 训练 hmm模型1
    print("----------------train_model_1--------------\n")
    new_hmm_1 = train(train_datas_hmm1,init_hmm,50)
    print(new_hmm_1)
    
    best_prob =  compute_prob_for_datas(model_hmm1,train_datas_hmm1)
    print("-----best_prob-------",best_prob)
    
    
    # 训练 hmm模型2
    print("----------------train_model_2--------------\n")
    new_hmm_2 = train(train_datas_hmm2,init_hmm,50)
    print(new_hmm_2)
    
    best_prob =  compute_prob_for_datas(model_hmm2,train_datas_hmm2)
    print("-----best_prob-------",best_prob)
    
    # 进行测试
    N_test1 = len(test_datas_hmm1)
    N_test2 = len(test_datas_hmm2)
    labs = [0 for i in range(N_test1)] + [1 for i in range(N_test2)]
    test_datas = test_datas_hmm1 + test_datas_hmm2
    
    for i,test_data in  enumerate(test_datas):
        
        score1 = forward(new_hmm_1,test_data)
        score2 = forward(new_hmm_2,test_data)
        det_lab = np.argmax([score1,score2])
        
        print("%f  %f  det_lab=%d  true_lab = %d"%(score1,score2,det_lab,labs[i])) 
        
    
    
    
    
    
    
    
    


    
    
    
    
    
    
   
    
    
                   