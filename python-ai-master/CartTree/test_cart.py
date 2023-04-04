import numpy as np
from cart import tree
from new_drawTree import PlotTree
import matplotlib.pyplot as plt
def indexSplit(N,train_ratio):
    N_train = int(N*train_ratio)
    index_random = np.random.permutation(N)
    index_train = index_random[:N_train]
    index_test = index_random[N_train:]

    return index_train,index_test


if __name__ == "__main__":
    # iris 数据处理
    file_data = 'iris.data'

    # 数据读取
    datas = np.loadtxt(file_data,dtype = float, delimiter = ',',usecols=(0,1,2,3))
    labs = np.loadtxt(file_data,dtype = str, delimiter = ',',usecols=(4))
    N,D = np.shape(datas)


    # 分为训练集和测试集和
    index_train,index_test = indexSplit(N,train_ratio=0.6)

  

    train_datas = datas[index_train,:]
    train_labs = labs[index_train]


    test_datas = datas[index_test,:]
    test_labs = labs[index_test]

    stopping_sz = 1

    decision_tree_classifier = tree( train_datas, train_labs,stopping_sz )
    decision_tree_classifier.fit()
    ret_tree = decision_tree_classifier.print_tree()
    print(ret_tree)

    # 树的绘制
   
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    ax = plt.subplot(111, frameon=False, **axprops)
    m_plotTree = PlotTree(ret_tree,ax=ax)
    m_plotTree.draw()






    # n_right =0
    # for i in range(test_datas.shape[0]):
    #     prediction = decision_tree_classifier.predict(test_datas[i])
        
    #     if prediction == test_labs[i]:
    #         n_right = n_right+1

        
    #     print(prediction,test_labs[i])

    # print("acc = %.2f%%"%(n_right*100/len(test_labs)))
 

