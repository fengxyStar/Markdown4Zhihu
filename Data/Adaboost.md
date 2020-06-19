@[TOC](从优化角度推导Adaboost)

# 1 回顾AdaBoost算法流程
　　Adaboost基本原理是将多个弱分类器加权组合，最终形成一个强分类器。算法中有两个重要的权重：**样本权重**和**分类器权重**。算法每一次迭代中只投入一个弱分类器进行训练，当前加权样本上计算得到误差，并通过误差来计算此分类器的权重，并更新样本的权重
## 1.1 模型参数
**样本**：$\{(x_1,y_1),(x_2,y_2)...(x_n,y_n)\}$，其中$y_i\in\{-1,1\}$

**分类器**：$\{k_1,k_2...k_l\}$，其中$k_j(x_i)\in\{-1,1\}$

**样本初始权重**：$w^{(1)}=(w^{(1)}_1,w^{(1)}_2...w^{(1)}_n)=(1,1,...,1)$

## 1.2  算法流程
假设我们进行T次迭代（**注**：迭代次数不一定要等于分类器数量），则

**对于 t = 1,2...,T**
1. 选择使得误差 $E_t$ 最小的分类器 k，并设其为$C_t$
$$E_t = \sum\limits^{n}_{i=1}\frac{w^{(t)}_i}{\sum\limits^{n}_{j=1}w^{(t)}_j}I\{y_i\neq C_t(x_i)\}$$
2. 通过误差可以计算该分类器的权重
$$a_t=\frac{1}{2}ln(\frac{1-E_t}{E_t})$$
3.   接下来更新样本权重
$$w^{(t+1)}_i=\frac{w^{(t)}_i}{\sum\limits^{n}_{j=1}w^{(t)}_j}exp\{2a_t I\{y_i\neq C_t(x_i)\}\}$$

**end**

## 1.3 算法结果
最终得到弱分类器的加权和
$$f(x_i)=\sum\limits^{T}_{t=1}a_tC_t(x_i)$$


# 2 从优化角度推导Adaboost
## 2.1 推导过程
### 2.1.1 第一次迭代
首先我们考虑Adaboost第一次迭代，对于分类器的选择和权重的确定，可以化为一个简单的优化问题：

$$y\in\{-1,1\}，设分类器f(x_i)=a_1C_1(x_i)，其中C_1(x_i)\in\{-1,1\}$$
$$损失函数： L(y_i,f(x_i))=e^{-y_if(x_i)}$$
$$总误差为：L(a_1,C_1)=\sum\limits^{n}_{i=1}e^{-y_if(x_i)}=\sum\limits^{n}_{i=1}e^{-y_ia_1C_1(x_i)}$$
则我们目标是求解优化问题：$(\hat{a_1},\hat{C_1})=argmin_{a_1,C_1}L(a_1,C_1)$

**证明：**
$L(a_1,C_1)=\sum\limits^{n}_{i=1}e^{-y_ia_1C_1(x_i)}$
　　　　　$=\sum\limits^{n}_{i=1}e^{-a_1}I\{y_i=C_1(x_i)\}+\sum\limits^{n}_{i=1}e^{a_1}I\{y_i\neq C_1(x_i)\}　(*)$
　　　　　$=\sum\limits^{n}_{i=1}e^{-a_1}I\{y_i=C_1(x_i)\}+\sum\limits^{n}_{i=1}e^{-a_1}I\{y_i\neq C_1(x_i)\}$
　　　　　　$-\sum\limits^{n}_{i=1}e^{-a_1}I\{y_i\neq C_1(x_i)\}+\sum\limits^{n}_{i=1}e^{a_1}I\{y_i\neq C_1(x_i)\}　(**)$
　　　　　$=\sum\limits^{n}_{i=1}e^{-a_1}+\sum\limits^{n}_{i=1}(e^{a_1}-e^{-a_1})I\{y_i\neq C_1(x_i)\}$
　　　　　$=ne^{-a_1}+(e^{a_1}-e^{-a_1})\sum\limits^{n}_{i=1}I\{y_i\neq C_1(x_i)\}$
　　　　　
所以 
$$\hat{C_1}=argmin_{C_1}\sum\limits^{n}_{i=1}I\{y_i\neq C_1(x_i)\}$$
此时分类器在样本上的误判率为
$$E_1=\frac{1}{n}\sum\limits^{n}_{i=1}I\{y_i\neq C_1(x_i)\}$$
从式子中可以看到，为了使得总体误差最小化，我们需要选择一个**使得误判率$E_1$最小的分类器**。

确定了 $\hat{C_1}$ 后，下面来求 $\hat{a_1}$，对 $L(a_1,C_1)$ 求一阶导
$$L(a_1,C_1)=n[e^{-a_1}+(e^{a_1}-e^{-a_1})E_1]$$
$$\frac{\partial L(a_1,C_1)}{\partial a_1}=n[-e^{-a_1}+(e^{a_1}+e^{-a_1})E_1]=0$$
$$\Rightarrow   -1+(e^{2a_1}+1)E_1=0$$
$$\Rightarrow  \hat{a_1}=\frac{1}{2}ln\frac{1-E_1}{E_1}$$
观察 $\hat{a_1}$，我们发现**当误判率 $E_1$ 越大时，$\hat{a_1}$ 越小**，这很符合我们的直观理解：**当这个弱分类器分类效果较差时，我们就给予它较小的权重，以减小对正确结果的影响**

　　　
### 2.1.2 第二次迭代
下面我们在第一个分类器的基础上，再加入一个弱分类器，对于这个分类器的选择和权重确定，同样是一个优化问题：

$$y\in\{-1,1\}，设分类器f(x_i)=\hat{a_1}\hat{C_1}(x_i)+a_2C_2(x_i)，其中C_2(x_i)\in\{-1,1\}$$
$$损失函数_{(没变)}： L(y_i,f(x_i))=e^{-y_if(x_i)}$$
$$总误差为：L(a_2,C_2)=\sum\limits^{n}_{i=1}e^{-y_if(x_i)}=\sum\limits^{n}_{i=1}e^{-y_i[\hat{a_1}\hat{C_1}(x_i)+a_2C_2(x_i)]}$$
则我们目标是求解优化问题：$(\hat{a_2},\hat{C_2})=argmin_{a_2,C_2}L(a_2,C_2)$

**(注：此时分类器中$\hat{a_1}$和$\hat{C_1}$是确定的，因为我们在2.1.1的第一步优化中已经把其解出了)**

**证明：**
$L(a_2,C_2)=\sum\limits^{n}_{i=1}e^{-y_i[\hat{a_1}\hat{C_1}(x_i)+a_2C_2(x_i)]}$
　　　　　$=\sum\limits^{n}_{i=1}e^{-y_i\hat{a_1}\hat{C_1}(x_i)}e^{-y_ia_2C_2(x_i)}$
　　　　　（设 $w_i=e^{-y_i\hat{a_1}\hat{C_1}(x_i)}$）
　　　　　$=\sum\limits^{n}_{i=1}w_ie^{-y_ia_2C_2(x_i)}$
　　　　　（对等式做2.1.1中$(*)(**)$中相同处理，这里只是多了一个$m_i$）
　　　　　$=\sum\limits^{n}_{i=1}w_ie^{-a_2}+\sum\limits^{n}_{i=1}w_i(e^{a_2}-e^{-a_2})I\{y_i\neq C_2(x_i)\}\}$
　　　　　$=e^{-a_2}\sum\limits^{n}_{i=1}w_i+(e^{a_2}-e^{-a_2})\sum\limits^{n}_{i=1}w_iI\{y_i\neq C_2(x_i)\}\}$
　　　　　
所以 
$$\hat{C_2}=argmin_{C_2}\sum\limits^{n}_{i=1}w_iI\{y_i\neq C_2(x_i)\}$$
此时分类器在样本上的误判率为
$$E_2=\frac{\sum\limits^{n}_{i=1}w_iI\{y_i\neq C_1(x_i)\}}{\sum\limits^{n}_{j=1}w_j}$$
$$（\sum\limits^{n}_{j=1}w_j为归一化常数，这里去掉该常数定义E_2也是可以的）$$
从式子中可以看到，在这一步优化中，为了使得总体误差最小化，我们需要选择一个**使得==加权==误判率$E_2$最小的分类器**。

**注意这里的加权二字，我们可以把$w_i$ 看作样本权重，其中**
$$w_i=e^{-y_i\hat{a_1}\hat{C_1}(x_i)}=\left\{\begin{matrix}  e^{-\hat{a_1}}　y_i=\hat{C_1}(x_i) \\e^{\hat{a_1}}　　y_i\neq\hat{C_1}(x_i)
\end{matrix}\right.$$

（其实在第一次迭代中 $w_i$ 也是存在的，不过我们默认其初始值为$\{1,1,...1\}$，所以没有特意写出）

**我们可以看到，$m_i$ 实际上是由第一个分类器决定的，它和大小 $\hat{C_1}$ 的分类效果有关：**
**1. 当$\hat{C_1}(x_i)$分类正确时，如果$\hat{C_1}$ 对所有样本的分类越准确， $\hat{a_i}$就越大，从而使得 $w_i$ 越小。**
　　　　　　　　　　　**如果$\hat{C_1}$ 对所有样本的分类越差， $\hat{a_i}$就越小，从而使得 $w_i$ 越大。**
**2. 当$\hat{C_1}(x_i)$分类错误时，如果$\hat{C_1}$ 对所有样本的分类越准确， $\hat{a_i}$就越大，从而使得 $w_i$ 越大。**
　　　　　　　　　　　**如果$\hat{C_1}$ 对所有样本的分类越差， $\hat{a_i}$就越小，从而使得 $w_i$ 越小。**　　　　
　　　　　　　　　　　　　　　　　
可以这样理解，好的分类器分错样本时，说明该样本容易分错，应增加权重，加强对其的训练。而差的分类器分类结果本来就很随意，分正确一个样本并没有太多意义，所以仍然要增强训练。

确定了 $\hat{C_2}$ 后，下面来求 $\hat{a_2}$，对 $L(a_2,C_2)$ 求一阶导
$$L(a_2,C_2)=\sum\limits^{n}_{i=1}w_i[e^{-a_2}+(e^{a_2}-e^{-a_2})E_2]$$
$$\frac{\partial L(a_2,C_2)}{\partial a_2}=\sum\limits^{n}_{i=1}w_i[-e^{-a_2}+(e^{a_2}+e^{-a_2})E_2]=0$$
$$\Rightarrow   -1+(e^{2a_2}+1)E_2=0$$
$$\Rightarrow  \hat{a_2}=\frac{1}{2}ln\frac{1-E_2}{E_2}$$
有了前两个分类器的参数，我们可以计算一下第三次迭代的**样本权重**
$$w_i^3=e^{-y_i[\hat{a_1}\hat{C_1}(x_i)+\hat{a_2}\hat{C_2}(x_i)]}=
w_i^2e^{-y_i\hat{a_2}\hat{C_2}(x_i)}$$
根据式子的规律，第k次迭代的样本权重就为
$$w_i^k=w_i^{k-1}e^{-y_i\hat{a_{k-1}}\hat{C_{k-1}}(x_i)}$$
### 2.1.2 第k次迭代
到了第k次迭代，我们已经得到了k-1个弱分类器加权组成的分类器，下面就要计算第k个分类器的各个参数
$$分类器为f(x_i)=\sum\limits^{k-1}_{t=1}\hat{a_t}\hat{C_t}(x_i)+a_k C_k(x_i)$$
1. 此次迭代的样本权重为(由前k-1个分类器决定) 
$$w_i^k=w_i^{k-1}e^{-y_i\hat{a_{k-1}}\hat{C_{k-1}}(x_i)}$$
2. 选择的分类器为
$$\hat{C_k}=argmin_{C_k}\sum\limits^{n}_{i=1}w_i^kI\{y_i\neq C_k(x_i)\}$$
其中误判率为
$$E_k=\frac{\sum\limits^{n}_{i=1}w_i^kI\{y_i\neq C_k(x_i)\}}{\sum\limits^{n}_{j=1}w_j^k}$$
（也就是选择使得加权误判率最小的分类器，可以回顾一下文章开始，adaboost算法的确是这么做的）

3. 计算该分类器的权重
$$a_k=\frac{1}{2}ln(\frac{1-E_k}{E_k})$$
4. 更新下一次迭代的样本权重
（步骤1实际上是k-1次迭代的步骤4，这里为了方便说明就都写上了）
$$w_i^{k+1}=w_i^{k}e^{-y_i\hat{a_{k}}\hat{C_{k}}(x_i)}$$
可以看到步骤2、3的公式都和文章开始的算法中的一致了，更新样本权重的公式有差别，但实际上是一样的，下面给出推导：
算法中的公式为
$$w^{(k+1)}_i=\frac{w^{(k)}_i}{\sum\limits^{n}_{j=1}w^{(k)}_j}exp\{2\hat{a_k} I\{y_i\neq \hat{C_k}(x_i)\}\}$$
我们推出的公式为
$$w_i^{k+1}=w_i^{k}e^{-y_i\hat{a_{k}}\hat{C_{k}}(x_i)}=w_i^{k}e^{-\hat{a_{k}}}e^{\hat{a_{k}}[1-y_i\hat{C_{k}}(x_i)]}$$
$$而　1-y_i\hat{C_{k}}(x_i) ＝\left\{\begin{matrix}  0　y_i=\hat{C_k}(x_i) \\2　y_i\neq\hat{C_k}(x_i)
\end{matrix}\right.=2I\{y_i\neq \hat{C_k}(x_i)\}$$
$$所以　 w_i^{k+1}=w_i^{k}e^{-\hat{a_{k}}}e^{2\hat{a_{k}}I\{y_i\neq \hat{C_k}(x_i)\}}$$
$$\propto \frac{w^{(k)}_i}{\sum\limits^{n}_{j=1}w^{(k)}_j}exp\{2\hat{a_k} I\{y_i\neq \hat{C_k}(x_i)\}\}$$
$$（\sum\limits^{n}_{j=1}w^{(k)}_j为归一化常数）$$


## 2.2 总结
 Adaboost 算法最终的的分类器是
 $$f(x_i)=\sum\limits^{T}_{t=1}a_tC_t(x_i)$$
 所以实际上Adaboost想优化的问题是
 $$\sum\limits^{n}_{i=1}L(y_i,f(x_i))$$
 
 但由于参数过多，直接优化容易导致过拟合问题，所以考虑加入惩罚项
 $$\sum\limits^{n}_{i=1}L(y_i,f(x_i))+\lambda_1 ||a||+\lambda_1 ||C||$$
 此优化问题非常复杂，难以求解，所以我们采用上面这种**每次优化一个分类器**的优化算法。该算法不仅解决了复杂性的问题，而且达到的结果是和上式是一致的。



