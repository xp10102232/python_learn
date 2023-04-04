import numpy as np

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
    
    
def gen_one_sample_from_GMM(gmm):
    # 根据ws 选取一个高斯成分
    k = gen_one_sample_from_Prob_list(gmm["ws"])
    
    mu = gmm["mus"][k]
    sigma = gmm["sigmas"][k]
    return np.random.multivariate_normal(mu, sigma)
    
    
    
def gen_samples_from_GMM_HMM(model,N):
    # M_O2S = model["M_O2S"]
    K,D = np.shape(model["S"][0]["mus"])
    datas = np.zeros([N,D])
    stats = np.zeros(N)
    
    # 得到初始状态，并根据初始状态生成一个样本
    init_S = gen_one_sample_from_Prob_list(model["pi"])
    stats[0] = init_S
    
    # 根据初始状态，生成一个数据
    datas[0] = gen_one_sample_from_GMM(model["S"][int(stats[0])]) 
    
    #生成其他样本 
    for i in range(1,N):
        # 根据前一状态，生成当前状态
        stats[i] = gen_one_sample_from_Prob_list(model["A"][int(stats[i-1])])
        # 根据当前状态生成一个数据
        datas[i] = gen_one_sample_from_GMM(model["S"][int(stats[i])])
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
    
    sum_gamma = np.sum(sum_xi,axis=1,keepdims=True)
    # for gamma in collect_gamma:
        # sum_gamma = sum_gamma+ np.sum(gamma[:-1],axis=0)

    # sum_gamma = np.tile(np.expand_dims(sum_gamma,axis=1),(1,N_stats))

    A = sum_xi/sum_gamma
    return A
    

            
def update_GMM_in_States(train_datas,model,collect_gamma):
    
    train_datas = np.concatenate(train_datas,axis=0)
    collect_gamma= np.concatenate(collect_gamma,axis=0)
    
    T,D = np.shape(train_datas)
    N_mix = np.shape(model["S"][0]["ws"])[0]
    N_state = len(model["S"])
    
    # 计算每个样本在每个状态的每个mix上的概率
    gamma_mix = np.zeros([T,N_state,N_mix])
    
    for t in range(T):
        # 样本在状态上的概率
        for s in range(N_state):
            # o 在状态 s 的每个 mixture上的概率
            p_mix = np.zeros(N_mix)
            for m in range(N_mix):
                p_mix[m] = getPdf(train_datas[t],model["S"][s]["mus"][m],model["S"][s]["sigmas"][m])
                p_mix[m] = p_mix[m]*model["S"][s]["ws"][m]
            p_mix = p_mix/np.sum(p_mix)
            gamma_mix[t,s,:] = p_mix*collect_gamma[t][s]
    
    # 进行参数的更新
    new_states = []
    for s in range(N_state):        
        gmm = dict()
        gmm["ws"] = np.zeros(N_mix)
        gmm["mus"] = np.zeros([N_mix,D])
        gmm["sigmas"] = np.zeros([N_mix,D,D])
        new_states.append(gmm)
    
    for s in range(N_state):
        for m in range(N_mix):
  
            r_k = gamma_mix[:,s,m]
            N_k = np.sum(r_k)
            r_k = r_k[:,np.newaxis] #[T,1]

            # 更新mu
            mu = np.sum(train_datas*r_k,axis=0)/ N_k #[D,1]
            
            # 更新sigma
            dx = train_datas - model["S"][s]["mus"][m]
            sigma = np.zeros([D,D])
            for t in range(T):
                sigma = sigma + r_k[t,0]*np.outer(dx[t],dx[t])
            sigma = sigma/N_k
            
            # 为sigma 加上一个比较小的对角线的值
            sigma = sigma + np.eye(D)*0.001
            # print("------sigma-----",sigma)
            # 更新 w
            w = N_k/T
            
            new_states[s]["mus"][m] = mu
            new_states[s]["sigmas"][m] = sigma
            new_states[s]["ws"][m] = w
        # 对 ws 进行正则
        new_states[s]["ws"] = new_states[s]["ws"]/np.sum(new_states[s]["ws"])
    return new_states
            
    
    
    
    
# 训练数据是一个numpy的列表
def train_step_GMM_HMM(train_datas,model):    
    
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
    new_states = update_GMM_in_States(train_datas,model,collect_gamma)

    return new_A,new_pi,new_states
    
    
def compute_prob_for_datas(model,datas):
    results = 0
    
    for o in datas:
        results += forward(model,o)
        
    return results
    
def train_GMM_HMM(train_datas,model,n_iteration):
    
    new_model = model.copy()
    
    prob_old = compute_prob_for_datas(new_model,train_datas)
    print("Prob_first",prob_old)
    
    for i in range(n_iteration):
        new_A,new_pi,new_states = train_step_GMM_HMM(train_datas,new_model)
        new_model["A"] =  new_A
        new_model["pi"] = new_pi
        new_model["S"]= new_states
        new_model["B"] = Pdf_O2GMMs
        
        prob_new =  compute_prob_for_datas(new_model,train_datas)
        print("it %d prob %f"%(i,prob_new))
      
        if prob_new>prob_old:
            prob_old = prob_new
            model = new_model
        else: 
            break
            
    return model  
    


def creat_GMM(mus,sigmas,ws):
    gmm = dict()
    gmm['mus'] = mus
    gmm['sigmas'] = sigmas
    gmm['ws'] = ws
    return gmm


# 计算一个高斯的pdf
# x: 数据 [D]
# sigma 方差 [D,D]
# mu 均值 [D]
def getPdf(x,mu,sigma):
    
    sigma = np.matrix(sigma)
    D = np.shape(x)[0]
    covar_det = np.linalg.det(sigma);
        
    c = (1 / ( (2.0*np.pi)**(float(D/2.0)) * (covar_det)**(0.5)))
    pdfval = c * np.exp(-0.5 * np.dot( np.dot((x-mu),sigma.I), (x-mu)) )
    return pdfval

# 计算GMM的pdf
def getPdfFromeGMM(x,gmm):
    K,D = np.shape(gmm["mus"])
    temp =0
    for k in range(K):
        temp += getPdf(x,gmm["mus"][k],gmm["sigmas"][k])*gmm["ws"][k]
    return temp

# 计算观测样本到每一个 状态（GMM）上的 pdf
def Pdf_O2GMMs(model,o):
    states = model["S"]
    N_states = len(states)
    pdfs = np.zeros(N_states)
    
    for i,gmm in enumerate(states):
        pdfs[i] = getPdfFromeGMM(o,gmm)
    return pdfs
        
   
    
    
    

if __name__ == "__main__":
    
    model_GMM_hmm1 = dict()
    
    model_GMM_hmm1["pi"] = np.array([1.0/3.0,1.0/3.0,1.0/3.0])
    
    model_GMM_hmm1["A"] = np.array([   
                        [0.8, 0.1, 0.1],
                        [0.8, 0.1, 0.1],
                        [0.8, 0.1, 0.1], 
                       ])
    states = []
    
    for i in range(3):
        # 3个高斯成分 K  特征维度是2  D
        mus = 0.6*np.random.random_sample([3,2])-0.3  #[K,D]
        sigmas= np.array([np.eye(2,2) for i in range(3)]) #[K,D,D]
        ws = np.array([0.3,0.5,0.2])                           #[K]  
        gmm = creat_GMM(mus,sigmas,ws)
        states.append(gmm)
        
    model_GMM_hmm1["S"] = states
    
    model_GMM_hmm1["B"] = Pdf_O2GMMs
    
        
   
    
  
    
    ''' 任务 0 数据生成'''
    datas,states = gen_samples_from_GMM_HMM(model_GMM_hmm1,50)
    print(datas)
    print(states)

        
    '''任务1 计算前向概率 '''
    obs = np.array([ [0.3,0.3], [0.1,0.1], [0.2,0.2]])
    prob_forward =forward(model_GMM_hmm1,obs)
    print(prob_forward)
    
    ''' 任务2 进行维特比译码  '''
    
    _,path = decoder(model_GMM_hmm1,obs)
    print(path)
    
   
    
    '''任务3 进行训练 '''
    # 构建GMM-HMM2
    model_GMM_hmm2 = dict()
    
    model_GMM_hmm2["pi"] = np.array([0.6,0.2,0.2])
    
    model_GMM_hmm2["A"] = np.array([   
                        [0.1, 0.1, 0.8],
                        [0.1, 0.8, 0.1],
                        [0.8, 0.1, 0.1], 
                       ])
    states = []
    
    for i in range(3):
        # 3个高斯成分 K  特征维度是2  D
        mus = 0.1*np.random.random_sample([3,2])-0.5  #[K,D]
        sigmas= np.array([np.eye(2,2) for i in range(3)]) #[K,D,D]
        ws = np.array([0.1,0.1,0.9])                          #[K]  
        gmm = creat_GMM(mus,sigmas,ws)
        states.append(gmm)
        
    model_GMM_hmm2["S"] = states
    model_GMM_hmm2["B"] = Pdf_O2GMMs
    
    
    
    # model_GMM_hmm1 生成一组训练、测试 数据
    train_datas_hmm1 = []
    for i in range(90):
        data_hmm1,_ = gen_samples_from_GMM_HMM(model_GMM_hmm1,50)
        train_datas_hmm1.append(data_hmm1)
    
    test_datas_hmm1 = []
    for i in range(10):
        data_hmm1,_ = gen_samples_from_GMM_HMM(model_GMM_hmm1,50)
        test_datas_hmm1.append(data_hmm1)
    
    # model_GMM_hmm2 生成一组训练、测试 数据
    train_datas_hmm2 = []
    for i in range(90):
        data_hmm2,_ = gen_samples_from_GMM_HMM(model_GMM_hmm2,50)
        train_datas_hmm2.append(data_hmm2)
        
    test_datas_hmm2 = []
    for i in range(10):
        data_hmm2,_ = gen_samples_from_GMM_HMM(model_GMM_hmm2,50)
        test_datas_hmm2.append(data_hmm2)
        
        
        
        
    
    # 初始化一个 GMM-HMM
    
    init_hmm = dict()
    N_state = 3
    N_mix =3
    D =2
    # 初始化 pi
    pi = np.random.random_sample(N_state)
    pi = pi / sum(pi)
    init_hmm["pi"] = pi
    
    # 初始化 A
    a = np.random.random_sample((N_state, N_state))
    row_sums = a.sum(axis=1)
    a = a / row_sums[:, np.newaxis]  
    init_hmm["A"] = a
    
    states = []
    
    for i in range(N_state):
        # 3个高斯成分 K  特征维度是2  D
        mus = np.random.random_sample([N_mix,D]) #[K,D]
        sigmas= np.array([np.eye(D,D) for i in range(3)]) #[K,D,D]
        ws = np.ones(N_mix)*(1.0/N_mix)                           #[K]  
        gmm = creat_GMM(mus,sigmas,ws)
        states.append(gmm)
        
    init_hmm["S"] = states
    init_hmm["B"] = Pdf_O2GMMs

    # 训练模型1
    new_gmm_hmm1 = train_GMM_HMM(train_datas_hmm1,init_hmm,50)
    
    # 训练模型2
    new_gmm_hmm2 = train_GMM_HMM(train_datas_hmm2,init_hmm,50)
    
    # 测试
    # 进行测试
    N_test1 = len(test_datas_hmm1)
    N_test2 = len(test_datas_hmm2)
    labs = [0 for i in range(N_test1)] + [1 for i in range(N_test2)]
    test_datas = test_datas_hmm1 + test_datas_hmm2
    
    for i,test_data in  enumerate(test_datas):
        score1 = forward(new_gmm_hmm1,test_data)
        score2 = forward(new_gmm_hmm2,test_data)
        det_lab = np.argmax([score1,score2])
        print("%f  %f  det_lab=%d  true_lab = %d"%(score1,score2,det_lab,labs[i])) 
        
    
    
    
    
    
   
    
    
                   