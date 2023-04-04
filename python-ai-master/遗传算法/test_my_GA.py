from hashlib import sha1
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from random import random,sample,choice,uniform

from math import floor
def data_generate():
    Points = []
    N = 100
    for i in range(N):
        X = uniform(-2.5, 2.5)
        Y = uniform(-2.5, 2.5)
        Points.append((X, Y))
    class_1 = []
    class_2 = []

    for point in Points:
        
        if point[1]-point[0]-0.5>0:
            class_1.append(point)
        elif point[1]-point[0]+0.5<0:
            class_2.append(point)

    N1 = len(class_1)
    N2 = len(class_2) 
    print(N1,N2)
    datas = class_1+class_2
    labs = [0]*N1+[1]*N2
    return np.array(datas),np.array(labs)

def my_draw_line(datas,labs,ws,n_cluster=2,str_title=""):
    plt.cla()
    
    colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1,n_cluster)]
    
    # 画点
    for i,lab in enumerate(labs):
            plt.scatter(datas[i,0],datas[i,1],s=16.,color=colors[lab-1])
    
    # 画线
    # 画判决线
    min_x = np.min(datas[:,0])
    max_x = np.max(datas[:,0])
  
    x = np.arange(min_x,max_x,0.01)
    
    for i,w in enumerate(ws):
        y = -(x*w[0]+w[2])/w[1]
        plt.plot(x,y,color=(0,0,0,1.0))

    w = ws[0]
    y = -(x*w[0]+w[2])/w[1]
    plt.plot(x,y,color=(1.0,0,0,1.0))
        
    plt.xlim(-2.5,2.5)
    plt.ylim(-2.5,2.5)

    plt.title(str_title)
    
    plt.show()

def sigmoid(z):
    return 1.0/(1+np.exp(-z))


def test_accuracy(datas,labs,w):
    
    if len(np.array(w).shape)!=2:
        w = np.array(w)
        w = np.expand_dims(w,axis=-1)

    N,D = np.shape(datas)
    z = np.dot(datas,w) # Nx1
    h = sigmoid(z)        # Nx1
    lab_det = (h>0.5).astype(np.float64)
    error_rate=np.sum(np.abs(labs-lab_det))/N
    return error_rate

def CE(X,Y,W):
    if len(np.array(W).shape)!=2:
        W = np.array(W)
        W = np.expand_dims(W,axis=-1)
    z = np.dot(X,W)
    h = sigmoid(z)
    eps= 0.000001
    loss_ce = -np.sum(Y*np.log(h+eps)+(1-Y)*np.log(1-h+eps)) 
    return loss_ce

def loss_SVM(datas,labs,w):
    if len(np.array(w).shape)!=2:
        w = np.array(w)
        w = np.expand_dims(w,axis=-1)
    labs_svm = labs.copy()
    labs_svm = labs_svm*2-1

    z = np.dot(datas,w) # Nx1
    loss_svm = np.abs(np.sum(z*labs_svm))/np.linalg.norm(w)
    return -loss_svm
     




# 制造一条样本  在这个就是随机制造一个 weight
# individual_size 样本的长度
# 在本例子中 就是weight 的长度 = 3
def create_individual(individual_size):
    """
    Create an individual.
    """
    return [random() for i in range(individual_size)]

# 制造一组人群  在这里就是制造一组 weight
# individual_size ： 每条样本的长度 这里是 3
# population_size： 构造了多少条样本  这里选择 50
def create_population(individual_size, population_size):
    """
    Create an initial population.
    """
    return [create_individual(individual_size) for i in range(population_size)]

# 计算损失函数
# 评估一条基因样本（individual）的好坏 
# datas labs 用来评估的数据
def get_fitness(individual, datas,labs):
    # 使用了三种评估手段
    
    # error 错误率
    error_rate = test_accuracy(datas,labs,individual)
    
    # ce_loss 交叉熵
    ce_loss = CE(datas,labs,individual)
    
    # svm_loss 
    svm_loss = loss_SVM(datas,labs,individual)
    
    # 三种损失 都是越小越好
    return {'CE': ce_loss, 
            'error': error_rate*100, 
            'svm':svm_loss,
            'coeff': individual}

# 对人群中的每条样本都进行评估，返回最好的样本

def evaluate_population(population,datas,labs,method,selection_size,best_individuals_stash):
    """
    Evaluate a population of individuals and return the best among them.
    """
    # 计算每条样本的适应度
    fitness_list = [get_fitness(individual, datas,labs)
                    for individual in population]
    
    # 以method 作为标准对 对获取的individuals 按照
    # 适应度从好到坏进行排序
    error_list = sorted(fitness_list, key=lambda i: i[method])
    
    #  选取最好的selection_size条
    best_individuals = error_list[: selection_size]

    # 将其中最好的一条先储存起来
    best_individuals_stash.append(best_individuals[0]['coeff'])

    # 将种群中最好的那条样本的适应度打印出来
    print('Error: ', best_individuals[0]['error'],
          'CE: ', best_individuals[0]['CE'],
          'svm',best_individuals[0]['svm'])
    
    # 返回最好的selection_size条样本
    return best_individuals

# 交叉融合
# 输入两条父母样本，通过基因交叉
# 生成一条子个体
def crossover(parent_1, parent_2):
   
    individual_size = len(parent_1['coeff'])

    loci = [i for i in range(0, individual_size)]
    # loci_1: 从parent_1 中选取的基因点位
    loci_1 = sample(loci, floor(0.5*(individual_size)))
    # loci_2 : 从parent_2 中选取的基因点位
    loci_2 = [i for i in loci if i not in loci_1]
    
    # 基因融合
    child = np.zeros(individual_size)
    child[loci_1] = np.array(parent_1['coeff'])[loci_1]
    child[loci_2] = np.array(parent_2['coeff'])[loci_2]
    return child.tolist()

   

# 基因突变
# 突变的概率 probability_of_gene_mutating
# 即，在个体中 有 多少个基因会突变
def mutate(individual,probability_of_gene_mutating):

    individual_size = len(individual)
    loci = [i for i in range(0, individual_size)]
    no_of_genes_mutated = floor(probability_of_gene_mutating*individual_size)
    # 随机选中突变的基因点
    loci_to_mutate = sample(loci, no_of_genes_mutated)
    
    for locus in loci_to_mutate:
        gene_transform = choice([-1, 1])
        change = gene_transform*random()
        # 对该基因点进行随机的加减
        individual[locus] = individual[locus] + change
    return individual


# 新的种群生成
# 利用选取的较好的基因生成新的种群
def get_new_generation(selected_individuals,population_size,
                       probability_of_individual_mutating,
                       probability_of_gene_mutating):

    # 在比较优秀的个体中中随机选择2个作为父母
    # 生成新的种群
    parent_pairs = [sample(selected_individuals, 2)
                    for i in range(population_size)]
    # 生成 population_size个 新的个体                
    offspring = [crossover(pair[0], pair[1]) for pair in parent_pairs]
    
    # 再从中选取部分进行基因突变
    offspring_indices = [i for i in range(population_size)]
    offspring_to_mutate = sample(
        offspring_indices,
        floor(probability_of_individual_mutating*population_size)
    )

    mutated_offspring = [[i, mutate(offspring[i],probability_of_gene_mutating)]
                         for i in offspring_to_mutate]
    for child in mutated_offspring:
        offspring[child[0]] = child[1]
    return offspring







if __name__ == "__main__":
    # 数据生成
    datas_in,labs_in = data_generate()
   
    # 对输入数据增加一个维度
    N,D = np.shape(datas_in)

    # 增加一个维度
    datas = np.c_[datas_in,np.ones([N,1])]

    # 对lab 进行维度调整 变为Nx1
    labs= np.expand_dims(labs_in,axis=-1) # Nx1


    # 遗传算法
    # 获取每条基因的长度
    individual_size = len(datas[0])
    
    # 繁衍种群的数目
    population_size = 1000

    # 从选取最好的10% 进行下一轮的种群繁衍
    rate_select = 0.2
    selection_size = floor(rate_select*population_size)
    
    # 繁衍的轮次
    max_generations = 50

    # 新生的种群中有 10% 的染色体会发生基因突变 
    probability_of_individual_mutating = 0.1
    
    # 每个发生突变的染色体中有100%的基因会发生变化 
    probability_of_gene_mutating = 1

    #best_possible = multiple_linear_regression(inputs, outputs)
    # 用来存储每轮最好的基因
    best_individuals_stash = [create_individual(individual_size)]
    
    # 种群初始化
    initial_population = create_population(individual_size, population_size)
    current_population = initial_population
    # termination = False
    generation_count = 0
    # 使用交叉熵作为 优质染色体选择的标准
    method = 'svm'
    for i in range(max_generations):
        plt.ion()
        current_best_individual = get_fitness(best_individuals_stash[-1], datas,labs)
        print('Generation: ', i)
        # 基因选择 从当前种群中选取最好的selection_size个染色体
        best_individuals = evaluate_population(current_population,
                                               datas,labs,method,
                                               selection_size,
                                               best_individuals_stash)
        # 使用交叉、突变的方法繁衍新的种群
        current_population = get_new_generation(best_individuals,
                                                population_size,
                                                probability_of_individual_mutating,
                                                probability_of_gene_mutating,
                                                )
        
        print(best_individuals_stash[-1])
        ws = [individual['coeff'] for individual in best_individuals]
        str_title="generate %d Error=%.1f CE =%.3f svm=%.1f"%(i,
                                                        best_individuals[0]['error'],
                                                        best_individuals[0]['CE'],
                                                        best_individuals[0]['svm'])
        my_draw_line(datas_in,labs_in,ws,n_cluster=2,str_title=str_title)
        plt.pause(0.5)
        plt.ioff()

    my_draw_line(datas_in,labs_in,[best_individuals_stash[-1]],n_cluster=2)
    plt.show()
    
 




















    # print(datas.shape)
    # print(labs)
    # my_draw(datas,labs)

