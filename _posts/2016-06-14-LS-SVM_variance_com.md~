---
layout: post
title: LS-SVM的各种变形对比
date: 2016-06-14
---


**分类**
$$
\begin{equation}
	\min_{w,b,e} J_{P}(w,e)=\frac{1}{2}w^{T}w+\gamma \frac{1}{2}\sum_{k=1}^{N}e_{k}^{2}       \\\
	s.t: y_{k}[w^{T}\varphi (x_{k})+b]=1-e_{k}, k=1,2,...,N
\end{equation}
$$

**回归**
$$
\begin{equation}
	\min_{w,b,e} J_{P}(w,e)=\frac{1}{2}w^{T}w+\gamma \frac{1}{2}\sum_{k=1}^{N}e_{k}^{2}       \\\
	s.t: y_{k}=w^{T}\varphi (x_{k})+b+e_{k}, k=1,2,...,N
\end{equation}
$$


**FDA**
思考FDA的本质, 是为了最大化类间距离, 最小化类内距离, 投影到一维平面.
最小化类内距离可以由min这个方程实现.
$$
\begin{equation}
	\min_{w,b,e} J_{P}(w,e)=\frac{1}{2}w^{T}w+\gamma \frac{1}{2}(\sum_{k=1}^{N1}e_{k}^{2}+\sum_{l=1}^{N2}e_{l}^{2})\\\
	y_{k}[w^{T}\varphi (x_{k})+b]=t_{1}-e_{k}, k=1,2,...,N1\\\
	y_{l}[w^{T}\varphi (x_{l})+b]=t_{2}-e_{l}, l=1,2,...,N2
\end{equation}
$$

**线性PCA**
注意在PCA中,我们的目标是最大化方差, 于是优化问题变成了max, 但我们又希望继续保持$w^{T}w$最小化(为啥咧?), 于是给它加负号.
$$
\begin{equation}
	\max_{w,e} J_{P}(w,e)= \gamma \frac{1}{2}\sum_{k=1}^{N}e_{k}^{2}-\frac{1}{2}w^{T}w\\\
	e_{k}=w^{T}x_{k}, k=1,2,...,N
\end{equation}
$$


**Kernel PCA**
$$
\begin{equation}
	\max_{w,e} J_{P}(w,e)= \gamma \frac{1}{2}\sum_{k=1}^{N}e_{k}^{2}-\frac{1}{2}w^{T}w\\\
	e_{k}=w^{T}(\varphi(x_{k})-\hat{\mu_{\varphi}}), k=1,2,...,N
\end{equation}
$$
这里面的$\mu_{\varphi}= \frac{1}{N} \sum_{k=1}^{N}\varphi(x_{k})$, 最后转化到对偶空间求解.


**Spectral Clustering**
谱聚类实际上就是kernel PCA加权重!!
$$
\begin{equation}
	\max_{w,e} J_{P}(w,e)= \gamma \frac{1}{2}\sum_{k=1}^{N}v_{k}e_{k}^{2}-\frac{1}{2}w^{T}w\\\
	e_{k}=w^{T}(\varphi(x_{k})-\hat{\mu_{\varphi}}), k=1,2,...,N
\end{equation}
$$

**Kernel CCA**
Canonical correlation analysis(CCA)是为了找到两个多维变量分布之间的最大Correlation.


**总结**
1. 把supervised, semi-supervised, unsupervised learning 结合在一起.
2. kernel method
3. 最小二乘支持向量机是*核心问题*, 是广义上的优化建模.
4. 需要链接基础理论, 算法和应用.
5. 是一种可靠的方法, numerically, computationally, statistically.
6. 最小二乘支持向量机优化问题可以想办法用线性方程组求解来求解.(但在Johan的书里写的是用迭代的共轭梯度法求解,注意必须要先把系统矩阵转化为正定的!)
7. 每一个问题都涉及到最小化$w^{T}w$, 目的是为了获得唯一解?



**特别注意**
对训练数据较少, 测试数据较多的情况, 适合在对偶空间求解. 
在10000个test data, 50 training data情况下, 原始空间$w$矩阵为10000维, 对偶空间$\alpha$仅50维.

对训练数据较多, 测试数据较少的情况, 适合在原空间求解, 因为对偶空间的kernel matrix巨大.
在1000000个training data, 20个test data的情况下, 原始空间权重矩阵为20维, 对偶空间为1000000维, kernel matrix为$1000000 \times 1000000$.
即large scale问题适合原始空间的SVM.
