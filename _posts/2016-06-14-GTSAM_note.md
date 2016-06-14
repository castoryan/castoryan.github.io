---
title: GTSAM笔记
notebook: yqrpsl's notebook
tags:SLAM
---


从单目标线性卡尔曼滤波开始考虑.
## 我们需要用到哪些矩阵?
*  state矩阵
*  system model矩阵
*  uncertainty矩阵P
*  观测矩阵

这些矩阵都是怎么排列的? 

## 如果推广到多目标?


## 如果是非线性?




# ISAM
整个KAESS的论文分三块:
* SAM: batch求解整个SLAM问题
* iSAM: 利用SLAM观测顺序进入的性质, 发现不是每次运算都需要从头开始. 于是只需要根据进入的观测, 部分更新矩阵. 可以加速. 利用reordering, relinearization.
* Data association: 利用JCBB来寻找好的数据关联. 数据关联是每次新观测值进入时必须进行的一步, 需要列举所有的关联, 然后测试哪一个关联最能够使目标函数最小化. 这个可以看Dellaert的论文.



原始问题:
$$\theta^{*}=argmin_{\theta}\|A\theta-b\|^{2}$$
一个最小二乘问题, 要求解则$\|A\theta-b\|^{2}$的导数为零, 得出$A^{T}A\theta=A^{T}b$. 可以通过Cholesky decomposition来求解$A^{T}A$.
$A$是measurement的Jacobian. 可以通过QR分解为:
$$A=Q[R;0]$$

$$
\begin{equation}
\|A\theta-b\|^{2}=\|Q[R;0]\theta-b\|^{2}\\\
=\|Q^{T}Q[R;0]\theta-Q^{T}b\|^{2}\\\
=\|[R;0]\theta-[d;e]\|^{2}\\\
=\|R\theta-d\|^{2}+\|e\|^{2}
\end{equation}
$$

* reordering
* relinearization
	- 论文P32中提到, 在isam中是把reordering和relinearization同时处理的. 
	- reordering是为了加快求解速度.
	- relinearization是为了处理非线性的问题, 改变线性化点.
	- 整个矩阵规模很大. 但是在SLAM中, 数据都是顺序进入的, 很多运算重复, 所以其实只有部分矩阵需要重新运算. 有一种办法把需要改变的部分识别出来, 然后只求那一部分.