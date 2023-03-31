# feature 选择

## 1. Filter
统计方法（卡方,方差） =》find feature与 label的相关性  
与 模型训练 完全无关  
## 2. embedded
评估 feature的重要程度(measurement) when 进行模型训练（why embedded）   
=》 按照 重要性 排序；  
=》 针对 重要性，for each feature 设置 权重(相关性得分
与 模型训练 相关   
eg，lasso regression（Lasso回归算法），decision tree（决策树）  
## 3. wrapper
按照 feature的组合，找到效果最好的  


# feature learning，representation learning

## 1. 基于 machine learning  =》学习 复杂数据的 best representation
what？ feed all feature，模型 会自动发现 feature的 重要程度，显著程度  
优势：减少了 业务专家的工作 =》难以 scale out  
eg， data：图片，视频  
eg，模型：Neural networks（神经网络）  

## 2. clustering, 聚类
what？ 发现 data 的 logical grouping  
eg，
when data：图片 in 无监督式学习； then 无监督式学习 中 的 feature learning:   dictionary learning (针对 密集features 有 sparse显著性)  
when deap learning（尤其 Neural networks）； then autoencoders（提取 数据的 显著性）  

# feature extraction
derived feature  

特点：  
1. 无法靠直觉意识到，难以解释  
2. 由 input features 转化而来  
   
范围：   
适用于 所有类型的数据  
eg  
图片：key points & descriptors  =》interesting eara  
数字：重新定义 new axes =》主特征（from 特征值，特征向量）  
文本（in 自然语言 处理）：Tf-Idf( term frequency inverse document frequency) =》 为 一个word in a document 的显著性 打分  

how?
1. dimensionality reduction: 降维


目标：  
1. 重新 以better的形式 组织data的 新feature from original feature；而非 降低feature的维度
   

# feature combination
why？  
1. input features 包含太多信息， some 与 label无关  
2. seperate features 的预测能力 < combined features 的预测能力

how？  
1. aggregating（features的组合） =》增强 预测能力（TODO: why？？？）  
eg：   
交通堵塞 预测：（day of week + time of day）
温度 预测：（season + time of day）
2. combined features 不仅仅是 sum of parts


# 降维（dimensionality reduction）
why？  
too much data（a curse and not a blessing）：the curse of dimensionality
1. hard to visualizing data
2. hard to training
3. hard to prediction =》machine learning models hard to find patterns from data =》poor quality models（overfiting：perform well in training， but poorly in the real word）

target  
1. pre-processing 算法 =》降低 raw features 的复杂度
2. 尤其， 减少 feature 的 数目 =》 fewer features

so
1. 解决 the curse of dimensionality
2. 同时 保留尽可能多 的有用信息 from underlying features（not lose too much useful information）

范围：
1. dimensionality reduction 属于 无监督式学习 的一种 =》无label 的data


how？
1. PCA( principle components analysis)，主成分分析 for linear data  =》reorienting original data： features 投影到 一个better axes（特征值，特征向量）
2. manifold learning for nonlinear data =》unrolling（展开） 数据的 complex form 到 a higher axis =》a simpler form with lower dimension  
eg： 展开地毯 =》 3维 立体 到 2维 平面
3. latent semantic analysis（潜在 语义分析）:  a topic modeling，dimension reduction technique  for text data
4. autoencoding for image => find efficient lower dimension 去 表示 image




# data： training + test（validation）
what？
1. 数据 拆分： 训练集（~80%） + 验证集，测试集（~20%）
   1. why？ train，evaluate 的数据 应该分开： 
   2. 不能 既是 运动员，又是 裁判员
2. 验证集： 验证，评估 模型的好坏程度？  
   1. =》防止 过拟合，提高 model robustness for 未见过的data  
   2. sanity-check for performance measure： model work well for 未见过的 data


02：34
   
why？


















# 监督式学习
eg 神经网络

# 无监督式学习
无label的data（代表 现实世界 大多的情况）  
方法： clustering  




# Q：
## 1. what is chi-square(卡方)
## 2. what is ANOVA(方差)
## 3. what is Neural networks（神经网络）？
## 4. what is dictionary learning ？
## 5. what is Autoencoders ？
## 6. what is overfitting，underfitting ？
## 7. what is PCA( principle components analysis)，主成分分析 ？
## 8. what is linear data， nonlinear data？
## 9. what is manifold learning？
## 10. what is latent semantic analysis（潜在 语义分析）？
## 11. what is autoencoding？
## 9. what is 
## 9. what is 
## 9. what is 
## 9. what is 
## 9. what is 
## 9. what is 
## 9. what is 
## 9. what is 
## 9. what is 
## 9. what is 