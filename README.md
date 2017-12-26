# IMpro
用的编译器是python3.6.5
调用的SVM库函数为scilearn中的SVC，惩罚系数为1，核函数为RBF
调用的神经网络库函数为google的tensorflow-CPU
训练集为比特币阿尔法交易网络，共24000余条边
测试集为比特币场外交易（OTC）网络，共360000余条边
数据集引自snap.stanford
使用的传播模型为MIA模型，出自Scalable Influence Maximization for Prevalent Viral
Marketing in Large-Scale Social Networks，相较于朴素的IC和LT模型快了近1000倍，但是仍然是O（k*n^2）的，不可扩展。

我的思路是，对于一张传播网络图，先在MIA模型下，用greedy算法算出100个近似解，然后将其标为seeds,其余点为ordinary，
然后用学习模型训练，得到一个判决器。此时来一个新的且与原网络相似的网络图，用判决器预测一批种子节点，后续便只在这些种子节点中进行greedy,效率很高
