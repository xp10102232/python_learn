import numpy as np

# 从datas 的第 ind_fea 维特征中获取所有可能得分割阈值
def get_possible_splits(datas , ind_fea ):
    feas =datas[:,ind_fea]
    feas = np.unique(feas)
    feas = np.sort(feas)
    splits =[]
    for i in range(len(feas)-1):
        th = (feas[i]+feas[i+1])/2
        splits.append(th)
    
    return np.array(splits)

def gini_impurity( labs ):

    unique_labs = np.unique(labs)
    gini = 0
    for lab in unique_labs:
        n_pos = np.where(labs==lab)[0].shape[0]
        prob_pos = n_pos/len(labs)
       
        gini += prob_pos**2
    
    gini = 1-gini
    return gini


# 利用 split 对 datas 的 ind_fea 维进行分割
# 计算该分割的基尼增益
def eval_split(datas,labs,ind_fea,split):
    mask = datas[:,ind_fea]<=split
    index_l = np.where(mask==1)[0]
    index_r = np.where(mask==0)[0]
    labs_l = labs[index_l]
    labs_r = labs[index_r]

    weight_left = float(len(labs_l)/len(labs))
    weight_right = 1- weight_left

    gini_parent = gini_impurity(labs)
    gini_left = gini_impurity(labs_l)
    gini_right = gini_impurity(labs_r)
    
    weighted_gini = gini_parent - (weight_left*gini_left + weight_right*gini_right)

    return weighted_gini

class node:
    
    def __init__(self, datas, labs, parent):
        self.parent = parent 
        self.datas = datas
        self.labs = labs
        
        # 当前节点的gini纯度
        self.gini = gini_impurity( self.labs )

        # tree nodes left and right 
        self.left = None
        self.right = None
        
        # 当前节点的分割条件
        self.splitting_ind_fea = None
        self.threshold = 0
        
        # set leaf parameters to None
        self.leaf = False
        self.label = None
        self.confidence = None

    # 设置当前节点的分割条件
    def set_splitting_criteria( self, ind_fea, threshold ):
        self.splitting_ind_fea = ind_fea
        self.threshold = threshold   

    # stopping_sz 剩下的数据小于stopping_sz 停止分割
    def is_leaf( self, stopping_sz ):
        if len(self.labs) <= stopping_sz or self.gini == 0.0:
            return True
        else:
            return False


    # 找到当前节点 最佳的分割 ind_fea 以及其相应的阈值
    def find_splitting_criterion( self ):
        
        max_score = -1.0

        best_ind_fea = None
        threshold = 0.0

        dim_fea = np.shape(self.datas)[-1]

        for i in range(dim_fea):
            splits = get_possible_splits( self.datas, i )

            for split in splits:
                split_score = eval_split( self.datas, self.labs, i, split)
                if split_score > max_score:
                    max_score = split_score
                    best_ind_fea = i
                    threshold = split

        return max_score, best_ind_fea, threshold
        

    # 对当前的节点进行分割
    def split( self, ind_fea, threshold ):
        
        mask = self.datas[:,ind_fea]<=threshold
        index_l = np.where(mask==1)[0]
        index_r = np.where(mask==0)[0]
        labs_l = self.labs[index_l]
        labs_r = self.labs[index_r]
        datas_l = self.datas[index_l,:]
        datas_r = self.datas[index_r,:]
        
        print("Splitting %d samples into %d and %d samples by %d th =%.2f"%(len(self.labs),len(labs_l),len(labs_r),ind_fea,threshold))
        
        left = node( datas_l , labs_l,self )
        right = node(datas_r , labs_r,self )

        return left, right
    
    # 将当前节点设为叶子节点
    def set_as_leaf ( self ):
        
        # set leaf parameters
        self.leaf = True
        # 设置该节点的标签为，所剩数据中标签最多的数据
        labs = self.labs.tolist()
        self.label =  max(labs,key=labs.count)
        n_pos= len(np.where(self.labs == self.label)[0])
        self.confidence = float( n_pos/len(self.labs))
        

class tree:

    def __init__( self, datas, labs ,stopping_sz ):

        self.root = None
        self.datas = datas
        self.labs = labs
        self.stopping_sz = stopping_sz
        self.dic_tree = {}
    def __build_tree( self, root ):
        
        # 如果是叶子节点则返回
        if root.is_leaf(self.stopping_sz):
            root.set_as_leaf()
            return

        # 找到最佳分割
        max_score, best_ind_fea, threshold = root.find_splitting_criterion()

        if best_ind_fea == None:
            return 

        # 设置分割条件
        root.set_splitting_criteria( best_ind_fea, threshold )

        # 对当前节点进行分割
        left, right = root.split( best_ind_fea, threshold )
        root.left = left
        root.right = right 
        
        self.__build_tree(root.left)
        self.__build_tree(root.right)
        return

    def fit( self ):
        if self.root == None:
            self.root = node( self.datas, self.labs, None )
            self.__build_tree(self.root)
        

    def predict ( self , sample ):

        current = self.root
        while ( not current.leaf ):
            # check for split criterion
            if sample[current.splitting_ind_fea] <= current.threshold:
                current = current.left
            else:
                current = current.right

        return current.label
    
    def __print_tree(self,root):
        
        if root.leaf:
            
            return(root.label)
        
        ret_Tree = {}
        str_root= 'dim%d th=%.2f'%(root.splitting_ind_fea, root.threshold)
        
        ret_Tree[str_root]={}

        # str_left = "dim %d<%.2f"%(root.splitting_ind_fea, root.threshold)
        # str_right = "dim %d>%.2f"%(root.splitting_ind_fea, root.threshold)
        str_left = "<%.2f"%( root.threshold)
        str_right = ">%.2f"%(root.threshold)
        ret_Tree[str_root][str_left] = self.__print_tree(root.left)
        ret_Tree[str_root][str_right] = self.__print_tree(root.right)
        
        return ret_Tree
        
    def print_tree(self):
        
        return self.__print_tree(self.root)

    
        
    





