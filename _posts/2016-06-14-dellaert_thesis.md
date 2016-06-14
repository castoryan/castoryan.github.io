---
title: Frank Dellaert Thesis总结
notebook: yqrpsl's notebook
tags:SLAM, CV
---

Using extracted features to solve geometric estimation problems induces a data-association problem, also known as the correspondence problem.

Isolating a particular solution for the correspondence problem can lead to biased estimates of the unknown parameters. This is illustrated quite nicely in the simple pose-estimation example we have been considering.

Below, in Section 1.6, I will show that this problem can be resolved in a principled manner.

This dissertation deals with multi-view, feature-based geometric estimation problems where the correspondence or data-association is unknown.
两帧之间找correspondences已经很困难, 在多帧之间找难度级别指数增加. 存在polynomial-time的算法, 找两个特征集之间的匹配, 但找多于两个特征集的最优匹配是NP-complete的. 这篇dissertation考虑了这个问题.

目前的特征匹配算法:
* 主要是考虑两帧或三帧之间的multiview constraints.
* 多数的cv算法都是被定位于一个pre-processing步骤, 所以只给出一个best correspondences, 会造成估计结果偏差. 从统计学的角度出发, optimal estimate应该是一个分布, 而不是一个值. 这样的话还可以增加prior knowledge.
* 用soft correspondaces来表达分布, 但是很难算, intractable. 用采样来近似, mutual exclusion constraint.



* chapter 2&4关于maximum lakelihood estimation


In fact, for this example the local maximum closest to the ML estimate for known correspondence is actually the less likely one if we do not know the correspondence.

* known correspondence的likelihood function只有一个peak, 而unknown有两个peak.(why??)

the likelihood function given unknown correspondence can be obtained by summing together all the individual likelihood functions for all possible correspondences.

似然值的大小直接由2D translation后特征点与实际obervation之间的差别决定.

unknown correspondence的似然方程可以通过积分所有possible correspondences的似然方程来获得.!! 所以可能会有多个peak.


考虑图1.4的情况, Image1中type1的feature有3个, type2有2个, Image2中type1有4个, type2有2个.
如果不考虑feature type, 那么correspondence的所有可能性是$C_{6}^{5} \times 5!=120$
如果考虑feature type, correspondence的所有可能性是$C_{2}^{2} \times 2! \times C_{4}^{3} \times 3!=2\times 4\times 6=48$.

所以每一种correspondence可以对应出一个似然方程来. 图里面看不出它们之间的峰值差别很大.


用EM算法可以把正确的correspondence和正确的translation算出来.
但问题在于
1. 可能的correspondences随特征数量指数增加, 也随参与的帧数指数增加.
2. 积分出来的likelihood方程可能很复杂, 很多Local minimum.

EM算法无法保证找到likelihood的全局最大, 用deterministic annealing解决.



**用贝叶斯公式, 就是在已知measurement和correspondence的情况下, 加上先验, 推断参数的MAP**见论文P51. 如果用概率公式表达的话, 需要把correspondence的所有可能性全部积分掉, 得到已知measurement, 推断参数的式子. 然而correspondences的可能性太多, 没法整...
于是EM算法把correspondences当做一个hidden variable, 

## Soft Correspondences as Marginal Probabilities
图1.10, 6行每行代表一个image2的feature, 5列每列代表一个image1的feature. 方块颜色越深代表它们之间match的概率越大.


## Sampling
计算soft correspondences的marginal probabilites, intracable. 用Markov chain monte carlo方法进行采样.



文章还是用最小化reprojection error的方式来解决的.


## Occlusion and Clutter
有时候即使特征还在, detector也会检测不到. 然后还可能会有spurious measurement. 这些都可以用概率来建模.


## Incorporating Appearance 
可以考虑加上特征的appearance, 这样可以大大减小可能的correspondances数量. 文中给出了详细的解决方式~~~




## 小结
这篇论文主要是针对SFM过程中特征匹配的问题. 两三帧之间直接用descriptor做匹配并不是一个好办法. 这篇文章的思路是, 穷举出所有可能的匹配, 然后一一算re-projection error, 看哪一个能给出最小的.
但是这个问题很复杂, 积分出每一个参数的marginal probability, 计算量很大. 需要用sampling的方法来降低运算量.
EM算法


*想想这样穷举, 跟RANSAC有什么区别, 优势劣势??*