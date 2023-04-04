from matplotlib import pyplot as plt
import numpy as np
#决策节点
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
# 叶节点
leafNode = dict(boxstyle="round4", fc="0.8")
# 箭头、分支
arrow_args = dict(arrowstyle="<-")

class PlotTree:
    def __init__(self,inTree,ax):

        # 获取树的宽度和高度
        self.inTree = inTree
        self.totalW = float(self._getNumLeafs(inTree))
        self.totalD = float(self._getTreeDepth(inTree))

        # 设置初始的x,y偏移量
        self.xOff = -0.5/self.totalW 
        self.yOff = 1.0
        self.ax = ax 


    def _getNumLeafs(self,myTree):
        numLeafs = 0
        keys = myTree.keys()
        firstStr = list(keys)[0]
        secondDict = myTree[firstStr]
        for key in secondDict.keys():
            if type(secondDict[key]).__name__=='dict':
                numLeafs += self._getNumLeafs(secondDict[key])
            else:   numLeafs +=1
        return numLeafs
    
    def _getTreeDepth(self,myTree):
        maxDepth = 0
        keys = list(myTree.keys())
        firstStr = keys[0]
        secondDict = myTree[firstStr]
        for key in secondDict.keys():
            if type(secondDict[key]).__name__=='dict':
                thisDepth = 1 + self._getTreeDepth(secondDict[key])
            else:   thisDepth = 1
            if thisDepth > maxDepth: 
                maxDepth = thisDepth
        return maxDepth
    
    def _plotNode(self,nodeTxt, centerPt, parentPt, nodeType):
        self.ax.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    

    def _plotMidText(self,cntrPt, parentPt, txtString):
        # 获取 cntrPt, parentPt 的中点
        xMid =  (parentPt[0]+cntrPt[0])/2.0
        yMid = (parentPt[1]+cntrPt[1])/2.0
        
        # 计算 cntrPt、 parentPt 连线与水平方向的夹角
        if parentPt[0]-cntrPt[0] ==0:
            theta =90   
        else:
            theta = np.arctan((parentPt[1]-cntrPt[1])/(parentPt[0]-cntrPt[0]))*180/np.pi
       
        self.ax.text(xMid, yMid, txtString, va="center", ha="center", rotation=theta)

    # myTree 当前树
    # parentPt 父节点的位置
    # nodeTxt 指向当前树的文字
    def _plotTree(self,myTree, parentPt, nodeTxt):
        # 获取当前树的所有叶子节点的数目，即当前树的宽度
        numLeafs = self._getNumLeafs(myTree)  
        
        keys = list(myTree.keys())
        firstStr = keys[0] 

        # 当前节点的位置应该在所有当前树的中间   
        cntrPt = (self.xOff + (1.0 + float(numLeafs))/2.0/self.totalW, self.yOff)
        self._plotMidText(cntrPt, parentPt, nodeTxt)
        plt.pause(1)
        self._plotNode(firstStr, cntrPt, parentPt, decisionNode)
        plt.pause(1)
        secondDict = myTree[firstStr]
        # 每画深一层 yOff减少
        self.yOff = self.yOff - 1.0/self.totalD
        for key in secondDict.keys():
            # 下一层是字典 画树
            if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
                self._plotTree(secondDict[key],cntrPt,str(key))        #recursion
            # 下一层是叶子
            else:
                # 每画一个叶子  xOff 增加
                self.xOff = self.xOff + 1.0/self.totalW
                self._plotNode(secondDict[key], (self.xOff, self.yOff), cntrPt, leafNode)
                plt.pause(1)
                self._plotMidText((self.xOff, self.yOff), cntrPt, str(key))
                plt.pause(1)
        # 返回一层 yOff增加
        self.yOff = self.yOff + 1.0/self.totalD

    def draw(self):
        self._plotTree(self.inTree, (0.5,1.0), '')
        plt.show()


if __name__ =="__main__":


    # fig = plt.figure()

    # plt.annotate("", xy=(1, 0), xytext=(0.5, 0.5),
    #         arrowprops=dict(arrowstyle="->"))
    
    # plt.text(0.75, 0.25, "yuhong", va="baseline", ha="center", rotation=-45+7)
    # plt.show()


    in_tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    cn_tree= {'是否眼干': {'干涩': '不配镜', '正常': {'是否散光': {'否': {'年龄': {'青年': '软镜片', '老年': {'近视/远视': {'远视': '软镜片', '近视': '不配镜'}}, '中年': '软镜片'}}, '是': {'近视/远视': {'远视': {'年龄': {'青年': '硬镜片', '老年': '不配镜', '中年': '不配镜'}}, '近视': '硬镜片'}}}}}}
    # 画布布局
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    ax = plt.subplot(111, frameon=False, **axprops)
    m_plotTree = PlotTree(cn_tree,ax=ax)
    m_plotTree.draw()













