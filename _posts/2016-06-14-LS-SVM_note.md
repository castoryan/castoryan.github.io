---
layout: post
title: LS-SVM笔记
date: 2016-06-14
---

LS-SVM笔记
这一章打算从LS-SVM的角度去看kernel PCA

------------

#最优化问题
在搞明白SVM整个求解推导的过程中,必须得用到很多最优化的基础知识, 在这顺便整理一下.
##无约束最优化

$$
\begin{equation}
	min_{x} f(x)
	\label{eq:当前公式名}
\end{equation}
$$

##带等号约束的最优化

$$
\begin{equation}
	min_{x} f(x)\\\
	s.t g_{i}(x)=0, i=1,2,...,n
\end{equation}
$$
用拉格朗日法
二维平面里等高线与曲线相切的位置. 主函数是等高线, 等号约束就是一条曲线.当且仅当两条线相切的时候, 才有可能取到极值. 切线处二者导数相等, 这就是求解的等式....


##带不等号约束的最优化
$$
\begin{equation}
	min_{x} f(x)\\\
	s.t g_{i}(x)=0, i=1,2,...,n\\\
		h_{k}(x)<0, k=1,2,...,m
\end{equation}
$$
KKT条件,是拉格朗日法的拓展.


-------------------
#基本SVM(分类,回归)
-------------------
##Classification

###分类 线性数据
Primal problem:
$$
\begin{equation}
	min_{w,b} J_{P}(w)=\frac{1}{2}w^{T}w       \\\
	y_{k}[w^{T}x_{k}+b]>1, k=1,2,...,N
\end{equation}
$$
dual problem:
$$
\begin{equation}
	max_{\alpha} J_{D}(\alpha)= \- \frac{1}{2}\sum_{k,l=1}^{N} y_{k}y{l}x_{k}^{T}x_{l}\alpha_{k}\alpha_{l}+\sum_{k=1}^{N}\alpha_{k} \\\
	\sum_{k=1}^{N}\alpha_{k}y_{k}=0
\end{equation}
$$

注意: 对偶问题中已经是找一个$\alpha$使$L(w,b;\alpha)$最大了, 而我们原本是要找$w,b$使$J_{P}$最小的.
注意在对偶问题中,把求$w$问题转化成了求$\alpha$. 有几点好处:
1. 有全局且唯一解:
与$\alpha$对应的矩阵必然是正定(特征值全大于0)或半正定的(即特征值全大于等于0). 
当它是正定, 解全局且唯一.
当他是半正定, 解全局却不一定唯一.
有可能发生$(w,b)$唯一但是$\alpha$不唯一,这表示$w$可以用更少的支持向量表达.
2. 稀疏性:
大多数$\alpha$都为0, 于是在得到分类器之后, 对新数据分类时:
$$ y(x)= sign \[\sum_{k=1}^{\sharp sv}\alpha_{k}y_{k}x_{k}^{T}x+b\]$$
3. 几何含义:
求解得到的支持向量一定离决策边界比较近.
4. 无参及有参的问题:
原始问题中的$w$与data数量不相关, 而对偶问题中的$\alpha$与data量正相关, 所以对于大数据量,选原始问题. 高维数据, 选对偶问题.
5. 可以很方便地引入核函数, 因为对偶问题中有求 $<\x,\x_{i}>$的过程.

###分类 非线性数据
对于线性不可分的数据, 需要在约束中加入一个惩罚变量$\xi$, 试边界变软, 并限制$\xi$大于0才行.
Primal problem:
$$
\begin{equation}
	min_{w,b} J_{P}(w)=\frac{1}{2}w^{T}w+c\sum_{k=1}^{N}\xi_{k} \\\
	y_{k}[w^{T}x_{k}+b]>1-\xi_{k}, k=1,2,...,N \\\
	\xi_{k} \geq 0
\end{equation}
$$

dual problem:
$$
\begin{equation}
	max_{\alpha} J_{D}(\alpha)=\-\frac{1}{2}\sum_{k,l=1}^{N} y_{k}y{l}x_{k}^{T}x_{l}\alpha_{k}\alpha_{l}+\sum_{k=1}^{N}\alpha_{k} \\\
	\sum_{k=1}^{N}\alpha_{k}y_{k}=0 \\\
	0 \leq \alpha_{k} \leq c, k=1,...,N.
\end{equation}
$$



##Regression

##回归 线性方程
$$
\begin{equation}
	f(x)=w^{T}+b
\end{equation}
$$
则误差方程可定义为:
$$
\begin{equation}
	R=\frac{1}{N} \sum_{k=1}^{N} \|y_{k}-w^{T}-b\|
\end{equation}
$$

线性SVM回归时, 性质与分类类似.
* 解全局且唯一
* 原始问题与对偶问题求解等价???
* 解依然有稀疏性
* 对偶问题的规模与输入空间维度独立,与数据量正相关.
* 

###回归 非线性方程


* 原始问题与对偶问题不再等价,因为$w$向量的维度上升至无穷.
* 使用RBF核函数的非线性回归问题中参数有三个$\sigma c \varepsilon$, 需要用**交叉验证法**或**VC Bound**来决定参数.
* 大多是用RBF kernel, P60有很多拓展版本.....



---------
#LS-SVM(最小二乘支持向量机)
这是支持向量机的另一种表达, 具有一些特殊的特性.

---------

##LS-SVM原始
主要是要把凸二次型优化问题转化为线性方程求解, 更方便.
Primal problem:
$$
\begin{equation}
	min_{w,b,e} J_{P}(w,e)=\frac{1}{2}w^{T}w+\gamma \frac{1}{2}\sum_{k=1}^{N}e_{k}^{2}       \\\
	y_{k}[w^{T}\varphi (x_{k})+b]=1-e_{k}, k=1,2,...,N
\end{equation}
$$
$y(x)=sign[w^{T}\varphi (x)+b]$, $\varphi(\cdot)$就是把特征映射到高维空间.
比起原版SVM, 这里有两点改变:
1. 约束变成了带等号的,而不是一个阈值. 用$e_{k}$来容错.
2. 在主方程中加入了最小化误差的目标.

于是在对偶空间中的分类器为:
$$ 
\begin{equation}
	y(x)= sign \[\sum_{k=1}^{N}\alpha_{k}y_{k}K(x,x_{k})+b\]
\end{equation}
$$

LS-SVM性质:
1. kernel选择
2. 全局唯一解
3. 核心为KKT问题
4. 缺少了支持向量稀疏性的优势
5. 有参无参
6. tuning的参数


-----------
##多类问题
$$ 
\begin{equation}
\begin{cases}
y^{(1)} = &sign\[ w^{(1)^{T}}\varphi^{(1)}(x)+b^{(1)}\]\cr 
y^{(2)} = &sign\[ w^{(2)^{T}}\varphi^{(2)}(x)+b^{(2)}\]\cr 
...	\cr 
y^{(n)} = &sign\[ w^{(n)^{T}}\varphi^{(n)}(x)+b^{(n)}\]\cr 
\end{cases}
\end{equation}
$$
这样相当于对应每一类有一组解向量....那不就是N个SVM的意思???好麻烦...

* 书上说,通常多类分类($n_{C}$)问题都是转化成$n_{y}$个二分类来做的,对于$C_{i}$中, 多类中每一类有一个独立的编码$[y_{i}^{(1)};y_{i}^{(2)};...;y_{i}^{(n_{y})}] \subset {-1,+1}^{n_{y}}$, 分配一个值:$i=1,...,n_{C}$. 
* 关于 1vsA, 1vs1的对比讨论, 在P81.

##与Fisher discriminant analysis 的关联
关于线性判别分析的一篇blog, 讲得比较清楚了. [http://www.cnblogs.com/jerrylead/archive/2011/04/21/2024384.html]

$$
\begin{equation}
z=f(x)=w^{T}x+b
\end{equation}
$$
数据按照$w$进行映射, 到一维向量$z$. 条件约束是:Rayleigh quotient(RQ),
$$
\begin{equation}
max_{w,b}J_{FD}(w,b)=\frac{[E[z^{(1)}-E[z^{(2)}]]]^{2}}{E{[z^{(1)}-E[z^{(1)}]]^{2}}+E{[z^{(2)}-E[z^{(2)}]]^{2}}}
\end{equation}
$$
注意这个分子就是类间距离, 分母就是类内距离.
RQ是求特征值问题(啥意思?). 把$z=w^{T}x+b$带入上式, 并对上式对$w$求导.
直接摆出结果:
$$
\begin{equation}
w=S_{w}^{-1}(\mu_{1}-\mu_{2})\\\
=\frac{\mu_{1}-\mu_{2}}{S_{1}+S_{2}}
\end{equation}
$$
$S_{i}$是类内方差. 这个式子貌似很好理解了.这样求出来的映射权重$w$就把原始数据映射到了一维空间上.

FDA是为了把多维向量投影到一维上, 然后再该维度上做判别. 常见的那幅图片, 两个二维高斯分布, 椭圆形. 找一个方向使他们均值距离大,且方差大的那个方向平行. 这样一个方向, 两类之间距离最大(就是均值距离,其实每个方向都一样), 类内距离最小(高斯椭圆方差小的那个方向, 即类内方差最小.)
**看起来类内方差最小这一点要求与PCA正好相反, PCA只有一个类, 要求类内方差最大.**
所以FDA是定两个类, +1和-1. 最小化方差, 

这里介绍FDA是为了证明LS-SVM与FDA类似, 用类似的方式最小化类内距离(不过是在高维特征空间上), 并依然通过最小化$\|w\|$来最大化类间距离. 具体推导见p85.


##LS-SVM的求解
主要有几种方法:
1. SOR(Successive Over-Relaxation)
2. CG(Conjugate Gradient)
3. GMRES(Generalized Minimal Residual)

* CG要求矩阵必须正定, 然而LS-SVM中偏置项$b$的存在使得模型矩阵非正定. 所以在求解前需要先把矩阵转化为正定的.

##LS-SVM回归

只有两个$\sigma \gamma$参数

##与其它几个类似概念的关系
RKHS, Gaussian Process, regularization network都在第三章里有所提及.


##LS-SVM的稀疏性
LS-SVM的稀疏性不如原版, 从$\alpha_{k}=\gamma e_{k}$便可看出(咋看出来的?).
但是我们可以改进它! 用pruning.


------------
#非监督学习PCA
-----------
##线性PCA
1. 算所有特征$x$均值.
2. 构造$x$的协方差矩阵，算协方差矩阵的特征值和特征向量.
3. 取出前n大的特征值对应的特征向量，去乘原来的$x$均值，即对数据进行降维.
注意此时的PCA是线性的。用神经网络的方法可以把PCA扩展到非线性。

------------
##PCA与SVM
PCA可以理解为把一组均值为0的data映射到使它们方差最大的一个空间上.
$$
\begin{equation}
	max_{w} Var(w^{T}x)=Cov(w^{T}x, w^{T}x) \leftrightarrow \frac{1}{N}\sum_{k=1}^{N}(w^{T}x_{k}^2)=w^{T}Cw
\label{eq:KPCA1}
\end{equation}
$$
这里,矩阵 $C=\frac{1}{N}\sum_{k=1}{N}x_{k}x_{k}^{T}$, 就是要找到合适的$w$去满足方差最大这个目标, 还有一个约束是$w^{T}w=1$, 于是得到带约束的优化问题:
$$
\begin{equation}
	L(w,\lambda)=\frac{1}{2}w^{T}Cw-\lambda(w^{T}w-1)
\end{equation}
$$
有$\lambda$这个算子,于是有$\frac{\partial L}{\partial w}=0$, $\frac{\partial L}{\partial \lambda}=0$, 并有特征值求解问题:
$$
\begin{equation}
	Cw=\lambda w
\end{equation}
$$
C是对称矩阵, 半正定矩阵, **特征值最大的特征向量有最大的方差**.

则公式\ref{eq:KPCA1}的左半部分可以换一种写法:
$$
\begin{equation}
	max_{w}\sum_{k=1}^{N}[0-w^{T}x_{k}]^{2}
\end{equation}
$$
在这里0表示的是单个目标值. 这里的推导与LS-SVM与Fisher discriminant analysis类似. 在做FDA时考虑的是+1和-1的二分类问题, 在这里PCA考虑的是把目标定为0. 但是FDA目标是最小化类内方差,最大化类间距离. 而PCA最大化类内方差.. 


$$
\begin{equation}
	max_{w,e}J_{P}(w,e)= \gamma \frac{1}{2}\sum_{k=1}^{N}e_{k}^{2}-\frac{1}{2}w^{T}w\\\
	such that: e_{k}=w^{T}x_{k}, k=1,...N.
\end{equation}
$$

这个方程就是线性PCA用LS-SVM形式的表达. 即最小化$w_{T}w$的同时, 最大化$e_{k}$, 这个误差项是原始数据$x_{k}$通过$w$映射到新的空间的表达.



##谱聚类(Spectal Clustering)
[http://blog.csdn.net/v_july_v/article/details/40738211]
谱聚类可视作与PCA类似的过程???
谱聚类是从图的角度来进行聚类. 把聚类看成一个图分割的问题, 每个样本是一个图节点, 带权重的边为样本之间的相似度. 我们的目标是把大图分成若干个小图, 使子图内的边权重和尽量大, 子图之间的边权重和尽量小.
**相当于先进行非线性降维, 然后再低维空间聚类**

原始算法:
1. 对$N$个数据构造邻接矩阵, 再求出它的拉普拉斯矩阵
2. 对拉普拉斯矩阵求特征值 
3. 取出前k个特征值(最小的那k个),及它对应的k个特征向量, 组成$N \times k$的低维矩阵
4. 对低维矩阵用K-means进行聚类, 获得的类别 


###LS-SVM形式的谱聚类
把k类聚类问题考虑为一个k-1的binary问题. 有$N_{tr}$个训练数据.
$$
\begin{equation}
	\min_{w^{(l)},e^{(l)},b_{l}} \frac{1}{2}w^{(l)^{T}}w^{(l)} - \frac{1}{2}\sum_{l=1}^{k-1}\gamma_{l}e^{(l)^{T}}Ve^{(l)}\\\
	e^{(l)}=\Phi w^{(l)}+b_{l}1_{N}
\end{equation}
$$
$e^{(l)}=[e_{1}^{(l)},...,e_{1}^{(l)}]^{T}$是数据点 $\{x_{i}\}_{i=1}^{N}$ 在特征空间中的映射.
$e_{i}^{(l)}=w^{(l)^{T}} \varphi (x_{i})+b_{l}, i=1,...,N$

方法:
1. 





-----------
#ELM极限学习机

正好看到群里讨论,就直接看了一发, 笔记备忘.
其实很简单, ELM就是一个单隐层的感知机. input layer到hidden layer有一组权重$W$, 每个hidden node都有一个激活函数$\phi$, hidden layer到activation layer有一组权重$M$, 每一个隐层节点各有一个bias项,有n项, 合起来是$B$.
则由公式表达: 
$$
\begin{equation}
	\sum_{l=1}^{n} m_{l}\phi (X\cdot W+b_{l})
\end{equation}
$$
现在的目标是输出输入的误差最小化
$$
\begin{equation}
	min (y-x)^2=min_{M} (\sum_{l=1}^{n} m_{l}\phi (X\cdot W+b_{l}) - X)
\end{equation}
$$

正常的神经网络是用优化法, 分别优化$W,B,M$三个标量.

ELM则随机初始化$W,B$, 然后直接线性求解$M$.必须快啊..然而结果如何可真是看运气了.
