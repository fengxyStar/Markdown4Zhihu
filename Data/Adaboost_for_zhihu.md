@[TOC](从优化角度推导Adaboost)

# 1 回顾AdaBoost算法流程
　　Adaboost基本原理是将多个弱分类器加权组合，最终形成一个强分类器。算法中有两个重要的权重：**样本权重**和**分类器权重**。算法每一次迭代中只投入一个弱分类器进行训练，当前加权样本上计算得到误差，并通过误差来计算此分类器的权重，并更新样本的权重
## 1.1 模型参数
**样本**： <img src="https://www.zhihu.com/equation?tex=\{(x_1,y_1),(x_2,y_2)...(x_n,y_n)\}" alt="\{(x_1,y_1),(x_2,y_2)...(x_n,y_n)\}" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex=y_i\in\{-1,1\}" alt="y_i\in\{-1,1\}" class="ee_img tr_noresize" eeimg="1"> 

**分类器**： <img src="https://www.zhihu.com/equation?tex=\{k_1,k_2...k_l\}" alt="\{k_1,k_2...k_l\}" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex=k_j(x_i)\in\{-1,1\}" alt="k_j(x_i)\in\{-1,1\}" class="ee_img tr_noresize" eeimg="1"> 

**样本初始权重**： <img src="https://www.zhihu.com/equation?tex=w^{(1)}=(w^{(1)}_1,w^{(1)}_2...w^{(1)}_n)=(1,1,...,1)" alt="w^{(1)}=(w^{(1)}_1,w^{(1)}_2...w^{(1)}_n)=(1,1,...,1)" class="ee_img tr_noresize" eeimg="1"> 

## 1.2  算法流程
假设我们进行T次迭代（**注**：迭代次数不一定要等于分类器数量），则

**对于 t = 1,2...,T**
1. 选择使得误差  <img src="https://www.zhihu.com/equation?tex=E_t" alt="E_t" class="ee_img tr_noresize" eeimg="1">  最小的分类器 k，并设其为 <img src="https://www.zhihu.com/equation?tex=C_t" alt="C_t" class="ee_img tr_noresize" eeimg="1"> 

<img src="https://www.zhihu.com/equation?tex=E_t = \sum\limits^{n}_{i=1}\frac{w^{(t)}_i}{\sum\limits^{n}_{j=1}w^{(t)}_j}I\{y_i\neq C_t(x_i)\}" alt="E_t = \sum\limits^{n}_{i=1}\frac{w^{(t)}_i}{\sum\limits^{n}_{j=1}w^{(t)}_j}I\{y_i\neq C_t(x_i)\}" class="ee_img tr_noresize" eeimg="1">
2. 通过误差可以计算该分类器的权重

<img src="https://www.zhihu.com/equation?tex=a_t=\frac{1}{2}ln(\frac{1-E_t}{E_t})" alt="a_t=\frac{1}{2}ln(\frac{1-E_t}{E_t})" class="ee_img tr_noresize" eeimg="1">
3.   接下来更新样本权重

<img src="https://www.zhihu.com/equation?tex=w^{(t+1)}_i=\frac{w^{(t)}_i}{\sum\limits^{n}_{j=1}w^{(t)}_j}exp\{2a_t I\{y_i\neq C_t(x_i)\}\}" alt="w^{(t+1)}_i=\frac{w^{(t)}_i}{\sum\limits^{n}_{j=1}w^{(t)}_j}exp\{2a_t I\{y_i\neq C_t(x_i)\}\}" class="ee_img tr_noresize" eeimg="1">

**end**

## 1.3 算法结果
最终得到弱分类器的加权和

<img src="https://www.zhihu.com/equation?tex=f(x_i)=\sum\limits^{T}_{t=1}a_tC_t(x_i)" alt="f(x_i)=\sum\limits^{T}_{t=1}a_tC_t(x_i)" class="ee_img tr_noresize" eeimg="1">


# 2 从优化角度推导Adaboost
## 2.1 推导过程
### 2.1.1 第一次迭代
首先我们考虑Adaboost第一次迭代，对于分类器的选择和权重的确定，可以化为一个简单的优化问题：


<img src="https://www.zhihu.com/equation?tex=y\in\{-1,1\}，设分类器f(x_i)=a_1C_1(x_i)，其中C_1(x_i)\in\{-1,1\}" alt="y\in\{-1,1\}，设分类器f(x_i)=a_1C_1(x_i)，其中C_1(x_i)\in\{-1,1\}" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=损失函数： L(y_i,f(x_i))=e^{-y_if(x_i)}" alt="损失函数： L(y_i,f(x_i))=e^{-y_if(x_i)}" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=总误差为：L(a_1,C_1)=\sum\limits^{n}_{i=1}e^{-y_if(x_i)}=\sum\limits^{n}_{i=1}e^{-y_ia_1C_1(x_i)}" alt="总误差为：L(a_1,C_1)=\sum\limits^{n}_{i=1}e^{-y_if(x_i)}=\sum\limits^{n}_{i=1}e^{-y_ia_1C_1(x_i)}" class="ee_img tr_noresize" eeimg="1">
则我们目标是求解优化问题： <img src="https://www.zhihu.com/equation?tex=(\hat{a_1},\hat{C_1})=argmin_{a_1,C_1}L(a_1,C_1)" alt="(\hat{a_1},\hat{C_1})=argmin_{a_1,C_1}L(a_1,C_1)" class="ee_img tr_noresize" eeimg="1"> 

**证明：**
 <img src="https://www.zhihu.com/equation?tex=L(a_1,C_1)=\sum\limits^{n}_{i=1}e^{-y_ia_1C_1(x_i)}" alt="L(a_1,C_1)=\sum\limits^{n}_{i=1}e^{-y_ia_1C_1(x_i)}" class="ee_img tr_noresize" eeimg="1"> 
　　　　　 <img src="https://www.zhihu.com/equation?tex==\sum\limits^{n}_{i=1}e^{-a_1}I\{y_i=C_1(x_i)\}+\sum\limits^{n}_{i=1}e^{a_1}I\{y_i\neq C_1(x_i)\}　(*)" alt="=\sum\limits^{n}_{i=1}e^{-a_1}I\{y_i=C_1(x_i)\}+\sum\limits^{n}_{i=1}e^{a_1}I\{y_i\neq C_1(x_i)\}　(*)" class="ee_img tr_noresize" eeimg="1"> 
　　　　　 <img src="https://www.zhihu.com/equation?tex==\sum\limits^{n}_{i=1}e^{-a_1}I\{y_i=C_1(x_i)\}+\sum\limits^{n}_{i=1}e^{-a_1}I\{y_i\neq C_1(x_i)\}" alt="=\sum\limits^{n}_{i=1}e^{-a_1}I\{y_i=C_1(x_i)\}+\sum\limits^{n}_{i=1}e^{-a_1}I\{y_i\neq C_1(x_i)\}" class="ee_img tr_noresize" eeimg="1"> 
　　　　　　 <img src="https://www.zhihu.com/equation?tex=-\sum\limits^{n}_{i=1}e^{-a_1}I\{y_i\neq C_1(x_i)\}+\sum\limits^{n}_{i=1}e^{a_1}I\{y_i\neq C_1(x_i)\}　(**)" alt="-\sum\limits^{n}_{i=1}e^{-a_1}I\{y_i\neq C_1(x_i)\}+\sum\limits^{n}_{i=1}e^{a_1}I\{y_i\neq C_1(x_i)\}　(**)" class="ee_img tr_noresize" eeimg="1"> 
　　　　　 <img src="https://www.zhihu.com/equation?tex==\sum\limits^{n}_{i=1}e^{-a_1}+\sum\limits^{n}_{i=1}(e^{a_1}-e^{-a_1})I\{y_i\neq C_1(x_i)\}" alt="=\sum\limits^{n}_{i=1}e^{-a_1}+\sum\limits^{n}_{i=1}(e^{a_1}-e^{-a_1})I\{y_i\neq C_1(x_i)\}" class="ee_img tr_noresize" eeimg="1"> 
　　　　　 <img src="https://www.zhihu.com/equation?tex==ne^{-a_1}+(e^{a_1}-e^{-a_1})\sum\limits^{n}_{i=1}I\{y_i\neq C_1(x_i)\}" alt="=ne^{-a_1}+(e^{a_1}-e^{-a_1})\sum\limits^{n}_{i=1}I\{y_i\neq C_1(x_i)\}" class="ee_img tr_noresize" eeimg="1"> 
　　　　　
所以 

<img src="https://www.zhihu.com/equation?tex=\hat{C_1}=argmin_{C_1}\sum\limits^{n}_{i=1}I\{y_i\neq C_1(x_i)\}" alt="\hat{C_1}=argmin_{C_1}\sum\limits^{n}_{i=1}I\{y_i\neq C_1(x_i)\}" class="ee_img tr_noresize" eeimg="1">
此时分类器在样本上的误判率为

<img src="https://www.zhihu.com/equation?tex=E_1=\frac{1}{n}\sum\limits^{n}_{i=1}I\{y_i\neq C_1(x_i)\}" alt="E_1=\frac{1}{n}\sum\limits^{n}_{i=1}I\{y_i\neq C_1(x_i)\}" class="ee_img tr_noresize" eeimg="1">
从式子中可以看到，为了使得总体误差最小化，我们需要选择一个**使得误判率 <img src="https://www.zhihu.com/equation?tex=E_1" alt="E_1" class="ee_img tr_noresize" eeimg="1"> 最小的分类器**。

确定了  <img src="https://www.zhihu.com/equation?tex=\hat{C_1}" alt="\hat{C_1}" class="ee_img tr_noresize" eeimg="1">  后，下面来求  <img src="https://www.zhihu.com/equation?tex=\hat{a_1}" alt="\hat{a_1}" class="ee_img tr_noresize" eeimg="1"> ，对  <img src="https://www.zhihu.com/equation?tex=L(a_1,C_1)" alt="L(a_1,C_1)" class="ee_img tr_noresize" eeimg="1">  求一阶导

<img src="https://www.zhihu.com/equation?tex=L(a_1,C_1)=n[e^{-a_1}+(e^{a_1}-e^{-a_1})E_1]" alt="L(a_1,C_1)=n[e^{-a_1}+(e^{a_1}-e^{-a_1})E_1]" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=\frac{\partial L(a_1,C_1)}{\partial a_1}=n[-e^{-a_1}+(e^{a_1}+e^{-a_1})E_1]=0" alt="\frac{\partial L(a_1,C_1)}{\partial a_1}=n[-e^{-a_1}+(e^{a_1}+e^{-a_1})E_1]=0" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=\Rightarrow   -1+(e^{2a_1}+1)E_1=0" alt="\Rightarrow   -1+(e^{2a_1}+1)E_1=0" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=\Rightarrow  \hat{a_1}=\frac{1}{2}ln\frac{1-E_1}{E_1}" alt="\Rightarrow  \hat{a_1}=\frac{1}{2}ln\frac{1-E_1}{E_1}" class="ee_img tr_noresize" eeimg="1">
观察  <img src="https://www.zhihu.com/equation?tex=\hat{a_1}" alt="\hat{a_1}" class="ee_img tr_noresize" eeimg="1"> ，我们发现**当误判率  <img src="https://www.zhihu.com/equation?tex=E_1" alt="E_1" class="ee_img tr_noresize" eeimg="1">  越大时， <img src="https://www.zhihu.com/equation?tex=\hat{a_1}" alt="\hat{a_1}" class="ee_img tr_noresize" eeimg="1">  越小**，这很符合我们的直观理解：**当这个弱分类器分类效果较差时，我们就给予它较小的权重，以减小对正确结果的影响**

　　　
### 2.1.2 第二次迭代
下面我们在第一个分类器的基础上，再加入一个弱分类器，对于这个分类器的选择和权重确定，同样是一个优化问题：


<img src="https://www.zhihu.com/equation?tex=y\in\{-1,1\}，设分类器f(x_i)=\hat{a_1}\hat{C_1}(x_i)+a_2C_2(x_i)，其中C_2(x_i)\in\{-1,1\}" alt="y\in\{-1,1\}，设分类器f(x_i)=\hat{a_1}\hat{C_1}(x_i)+a_2C_2(x_i)，其中C_2(x_i)\in\{-1,1\}" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=损失函数_{(没变)}： L(y_i,f(x_i))=e^{-y_if(x_i)}" alt="损失函数_{(没变)}： L(y_i,f(x_i))=e^{-y_if(x_i)}" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=总误差为：L(a_2,C_2)=\sum\limits^{n}_{i=1}e^{-y_if(x_i)}=\sum\limits^{n}_{i=1}e^{-y_i[\hat{a_1}\hat{C_1}(x_i)+a_2C_2(x_i)]}" alt="总误差为：L(a_2,C_2)=\sum\limits^{n}_{i=1}e^{-y_if(x_i)}=\sum\limits^{n}_{i=1}e^{-y_i[\hat{a_1}\hat{C_1}(x_i)+a_2C_2(x_i)]}" class="ee_img tr_noresize" eeimg="1">
则我们目标是求解优化问题： <img src="https://www.zhihu.com/equation?tex=(\hat{a_2},\hat{C_2})=argmin_{a_2,C_2}L(a_2,C_2)" alt="(\hat{a_2},\hat{C_2})=argmin_{a_2,C_2}L(a_2,C_2)" class="ee_img tr_noresize" eeimg="1"> 

**(注：此时分类器中 <img src="https://www.zhihu.com/equation?tex=\hat{a_1}" alt="\hat{a_1}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\hat{C_1}" alt="\hat{C_1}" class="ee_img tr_noresize" eeimg="1"> 是确定的，因为我们在2.1.1的第一步优化中已经把其解出了)**

**证明：**
 <img src="https://www.zhihu.com/equation?tex=L(a_2,C_2)=\sum\limits^{n}_{i=1}e^{-y_i[\hat{a_1}\hat{C_1}(x_i)+a_2C_2(x_i)]}" alt="L(a_2,C_2)=\sum\limits^{n}_{i=1}e^{-y_i[\hat{a_1}\hat{C_1}(x_i)+a_2C_2(x_i)]}" class="ee_img tr_noresize" eeimg="1"> 
　　　　　 <img src="https://www.zhihu.com/equation?tex==\sum\limits^{n}_{i=1}e^{-y_i\hat{a_1}\hat{C_1}(x_i)}e^{-y_ia_2C_2(x_i)}" alt="=\sum\limits^{n}_{i=1}e^{-y_i\hat{a_1}\hat{C_1}(x_i)}e^{-y_ia_2C_2(x_i)}" class="ee_img tr_noresize" eeimg="1"> 
　　　　　（设  <img src="https://www.zhihu.com/equation?tex=w_i=e^{-y_i\hat{a_1}\hat{C_1}(x_i)}" alt="w_i=e^{-y_i\hat{a_1}\hat{C_1}(x_i)}" class="ee_img tr_noresize" eeimg="1"> ）
　　　　　 <img src="https://www.zhihu.com/equation?tex==\sum\limits^{n}_{i=1}w_ie^{-y_ia_2C_2(x_i)}" alt="=\sum\limits^{n}_{i=1}w_ie^{-y_ia_2C_2(x_i)}" class="ee_img tr_noresize" eeimg="1"> 
　　　　　（对等式做2.1.1中 <img src="https://www.zhihu.com/equation?tex=(*)(**)" alt="(*)(**)" class="ee_img tr_noresize" eeimg="1"> 中相同处理，这里只是多了一个 <img src="https://www.zhihu.com/equation?tex=m_i" alt="m_i" class="ee_img tr_noresize" eeimg="1"> ）
　　　　　 <img src="https://www.zhihu.com/equation?tex==\sum\limits^{n}_{i=1}w_ie^{-a_2}+\sum\limits^{n}_{i=1}w_i(e^{a_2}-e^{-a_2})I\{y_i\neq C_2(x_i)\}\}" alt="=\sum\limits^{n}_{i=1}w_ie^{-a_2}+\sum\limits^{n}_{i=1}w_i(e^{a_2}-e^{-a_2})I\{y_i\neq C_2(x_i)\}\}" class="ee_img tr_noresize" eeimg="1"> 
　　　　　 <img src="https://www.zhihu.com/equation?tex==e^{-a_2}\sum\limits^{n}_{i=1}w_i+(e^{a_2}-e^{-a_2})\sum\limits^{n}_{i=1}w_iI\{y_i\neq C_2(x_i)\}\}" alt="=e^{-a_2}\sum\limits^{n}_{i=1}w_i+(e^{a_2}-e^{-a_2})\sum\limits^{n}_{i=1}w_iI\{y_i\neq C_2(x_i)\}\}" class="ee_img tr_noresize" eeimg="1"> 
　　　　　
所以 

<img src="https://www.zhihu.com/equation?tex=\hat{C_2}=argmin_{C_2}\sum\limits^{n}_{i=1}w_iI\{y_i\neq C_2(x_i)\}" alt="\hat{C_2}=argmin_{C_2}\sum\limits^{n}_{i=1}w_iI\{y_i\neq C_2(x_i)\}" class="ee_img tr_noresize" eeimg="1">
此时分类器在样本上的误判率为

<img src="https://www.zhihu.com/equation?tex=E_2=\frac{\sum\limits^{n}_{i=1}w_iI\{y_i\neq C_1(x_i)\}}{\sum\limits^{n}_{j=1}w_j}" alt="E_2=\frac{\sum\limits^{n}_{i=1}w_iI\{y_i\neq C_1(x_i)\}}{\sum\limits^{n}_{j=1}w_j}" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=（\sum\limits^{n}_{j=1}w_j为归一化常数，这里去掉该常数定义E_2也是可以的）" alt="（\sum\limits^{n}_{j=1}w_j为归一化常数，这里去掉该常数定义E_2也是可以的）" class="ee_img tr_noresize" eeimg="1">
从式子中可以看到，在这一步优化中，为了使得总体误差最小化，我们需要选择一个**使得==加权==误判率 <img src="https://www.zhihu.com/equation?tex=E_2" alt="E_2" class="ee_img tr_noresize" eeimg="1"> 最小的分类器**。

**注意这里的加权二字，我们可以把 <img src="https://www.zhihu.com/equation?tex=w_i" alt="w_i" class="ee_img tr_noresize" eeimg="1">  看作样本权重，其中**

<img src="https://www.zhihu.com/equation?tex=w_i=e^{-y_i\hat{a_1}\hat{C_1}(x_i)}=\left\{\begin{matrix}  e^{-\hat{a_1}}　y_i=\hat{C_1}(x_i) \\e^{\hat{a_1}}　　y_i\neq\hat{C_1}(x_i)
\end{matrix}\right." alt="w_i=e^{-y_i\hat{a_1}\hat{C_1}(x_i)}=\left\{\begin{matrix}  e^{-\hat{a_1}}　y_i=\hat{C_1}(x_i) \\e^{\hat{a_1}}　　y_i\neq\hat{C_1}(x_i)
\end{matrix}\right." class="ee_img tr_noresize" eeimg="1">

（其实在第一次迭代中  <img src="https://www.zhihu.com/equation?tex=w_i" alt="w_i" class="ee_img tr_noresize" eeimg="1">  也是存在的，不过我们默认其初始值为 <img src="https://www.zhihu.com/equation?tex=\{1,1,...1\}" alt="\{1,1,...1\}" class="ee_img tr_noresize" eeimg="1"> ，所以没有特意写出）

**我们可以看到， <img src="https://www.zhihu.com/equation?tex=m_i" alt="m_i" class="ee_img tr_noresize" eeimg="1">  实际上是由第一个分类器决定的，它和大小  <img src="https://www.zhihu.com/equation?tex=\hat{C_1}" alt="\hat{C_1}" class="ee_img tr_noresize" eeimg="1">  的分类效果有关：**
**1. 当 <img src="https://www.zhihu.com/equation?tex=\hat{C_1}(x_i)" alt="\hat{C_1}(x_i)" class="ee_img tr_noresize" eeimg="1"> 分类正确时，如果 <img src="https://www.zhihu.com/equation?tex=\hat{C_1}" alt="\hat{C_1}" class="ee_img tr_noresize" eeimg="1">  对所有样本的分类越准确，  <img src="https://www.zhihu.com/equation?tex=\hat{a_i}" alt="\hat{a_i}" class="ee_img tr_noresize" eeimg="1"> 就越大，从而使得  <img src="https://www.zhihu.com/equation?tex=w_i" alt="w_i" class="ee_img tr_noresize" eeimg="1">  越小。**
　　　　　　　　　　　**如果 <img src="https://www.zhihu.com/equation?tex=\hat{C_1}" alt="\hat{C_1}" class="ee_img tr_noresize" eeimg="1">  对所有样本的分类越差，  <img src="https://www.zhihu.com/equation?tex=\hat{a_i}" alt="\hat{a_i}" class="ee_img tr_noresize" eeimg="1"> 就越小，从而使得  <img src="https://www.zhihu.com/equation?tex=w_i" alt="w_i" class="ee_img tr_noresize" eeimg="1">  越大。**
**2. 当 <img src="https://www.zhihu.com/equation?tex=\hat{C_1}(x_i)" alt="\hat{C_1}(x_i)" class="ee_img tr_noresize" eeimg="1"> 分类错误时，如果 <img src="https://www.zhihu.com/equation?tex=\hat{C_1}" alt="\hat{C_1}" class="ee_img tr_noresize" eeimg="1">  对所有样本的分类越准确，  <img src="https://www.zhihu.com/equation?tex=\hat{a_i}" alt="\hat{a_i}" class="ee_img tr_noresize" eeimg="1"> 就越大，从而使得  <img src="https://www.zhihu.com/equation?tex=w_i" alt="w_i" class="ee_img tr_noresize" eeimg="1">  越大。**
　　　　　　　　　　　**如果 <img src="https://www.zhihu.com/equation?tex=\hat{C_1}" alt="\hat{C_1}" class="ee_img tr_noresize" eeimg="1">  对所有样本的分类越差，  <img src="https://www.zhihu.com/equation?tex=\hat{a_i}" alt="\hat{a_i}" class="ee_img tr_noresize" eeimg="1"> 就越小，从而使得  <img src="https://www.zhihu.com/equation?tex=w_i" alt="w_i" class="ee_img tr_noresize" eeimg="1">  越小。**　　　　
　　　　　　　　　　　　　　　　　
可以这样理解，好的分类器分错样本时，说明该样本容易分错，应增加权重，加强对其的训练。而差的分类器分类结果本来就很随意，分正确一个样本并没有太多意义，所以仍然要增强训练。

确定了  <img src="https://www.zhihu.com/equation?tex=\hat{C_2}" alt="\hat{C_2}" class="ee_img tr_noresize" eeimg="1">  后，下面来求  <img src="https://www.zhihu.com/equation?tex=\hat{a_2}" alt="\hat{a_2}" class="ee_img tr_noresize" eeimg="1"> ，对  <img src="https://www.zhihu.com/equation?tex=L(a_2,C_2)" alt="L(a_2,C_2)" class="ee_img tr_noresize" eeimg="1">  求一阶导

<img src="https://www.zhihu.com/equation?tex=L(a_2,C_2)=\sum\limits^{n}_{i=1}w_i[e^{-a_2}+(e^{a_2}-e^{-a_2})E_2]" alt="L(a_2,C_2)=\sum\limits^{n}_{i=1}w_i[e^{-a_2}+(e^{a_2}-e^{-a_2})E_2]" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=\frac{\partial L(a_2,C_2)}{\partial a_2}=\sum\limits^{n}_{i=1}w_i[-e^{-a_2}+(e^{a_2}+e^{-a_2})E_2]=0" alt="\frac{\partial L(a_2,C_2)}{\partial a_2}=\sum\limits^{n}_{i=1}w_i[-e^{-a_2}+(e^{a_2}+e^{-a_2})E_2]=0" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=\Rightarrow   -1+(e^{2a_2}+1)E_2=0" alt="\Rightarrow   -1+(e^{2a_2}+1)E_2=0" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=\Rightarrow  \hat{a_2}=\frac{1}{2}ln\frac{1-E_2}{E_2}" alt="\Rightarrow  \hat{a_2}=\frac{1}{2}ln\frac{1-E_2}{E_2}" class="ee_img tr_noresize" eeimg="1">
有了前两个分类器的参数，我们可以计算一下第三次迭代的**样本权重**

<img src="https://www.zhihu.com/equation?tex=w_i^3=e^{-y_i[\hat{a_1}\hat{C_1}(x_i)+\hat{a_2}\hat{C_2}(x_i)]}=
w_i^2e^{-y_i\hat{a_2}\hat{C_2}(x_i)}" alt="w_i^3=e^{-y_i[\hat{a_1}\hat{C_1}(x_i)+\hat{a_2}\hat{C_2}(x_i)]}=
w_i^2e^{-y_i\hat{a_2}\hat{C_2}(x_i)}" class="ee_img tr_noresize" eeimg="1">
根据式子的规律，第k次迭代的样本权重就为

<img src="https://www.zhihu.com/equation?tex=w_i^k=w_i^{k-1}e^{-y_i\hat{a_{k-1}}\hat{C_{k-1}}(x_i)}" alt="w_i^k=w_i^{k-1}e^{-y_i\hat{a_{k-1}}\hat{C_{k-1}}(x_i)}" class="ee_img tr_noresize" eeimg="1">
### 2.1.2 第k次迭代
到了第k次迭代，我们已经得到了k-1个弱分类器加权组成的分类器，下面就要计算第k个分类器的各个参数

<img src="https://www.zhihu.com/equation?tex=分类器为f(x_i)=\sum\limits^{k-1}_{t=1}\hat{a_t}\hat{C_t}(x_i)+a_k C_k(x_i)" alt="分类器为f(x_i)=\sum\limits^{k-1}_{t=1}\hat{a_t}\hat{C_t}(x_i)+a_k C_k(x_i)" class="ee_img tr_noresize" eeimg="1">
1. 此次迭代的样本权重为(由前k-1个分类器决定) 

<img src="https://www.zhihu.com/equation?tex=w_i^k=w_i^{k-1}e^{-y_i\hat{a_{k-1}}\hat{C_{k-1}}(x_i)}" alt="w_i^k=w_i^{k-1}e^{-y_i\hat{a_{k-1}}\hat{C_{k-1}}(x_i)}" class="ee_img tr_noresize" eeimg="1">
2. 选择的分类器为

<img src="https://www.zhihu.com/equation?tex=\hat{C_k}=argmin_{C_k}\sum\limits^{n}_{i=1}w_i^kI\{y_i\neq C_k(x_i)\}" alt="\hat{C_k}=argmin_{C_k}\sum\limits^{n}_{i=1}w_i^kI\{y_i\neq C_k(x_i)\}" class="ee_img tr_noresize" eeimg="1">
其中误判率为

<img src="https://www.zhihu.com/equation?tex=E_k=\frac{\sum\limits^{n}_{i=1}w_i^kI\{y_i\neq C_k(x_i)\}}{\sum\limits^{n}_{j=1}w_j^k}" alt="E_k=\frac{\sum\limits^{n}_{i=1}w_i^kI\{y_i\neq C_k(x_i)\}}{\sum\limits^{n}_{j=1}w_j^k}" class="ee_img tr_noresize" eeimg="1">
（也就是选择使得加权误判率最小的分类器，可以回顾一下文章开始，adaboost算法的确是这么做的）

3. 计算该分类器的权重

<img src="https://www.zhihu.com/equation?tex=a_k=\frac{1}{2}ln(\frac{1-E_k}{E_k})" alt="a_k=\frac{1}{2}ln(\frac{1-E_k}{E_k})" class="ee_img tr_noresize" eeimg="1">
4. 更新下一次迭代的样本权重
（步骤1实际上是k-1次迭代的步骤4，这里为了方便说明就都写上了）

<img src="https://www.zhihu.com/equation?tex=w_i^{k+1}=w_i^{k}e^{-y_i\hat{a_{k}}\hat{C_{k}}(x_i)}" alt="w_i^{k+1}=w_i^{k}e^{-y_i\hat{a_{k}}\hat{C_{k}}(x_i)}" class="ee_img tr_noresize" eeimg="1">
可以看到步骤2、3的公式都和文章开始的算法中的一致了，更新样本权重的公式有差别，但实际上是一样的，下面给出推导：
算法中的公式为

<img src="https://www.zhihu.com/equation?tex=w^{(k+1)}_i=\frac{w^{(k)}_i}{\sum\limits^{n}_{j=1}w^{(k)}_j}exp\{2\hat{a_k} I\{y_i\neq \hat{C_k}(x_i)\}\}" alt="w^{(k+1)}_i=\frac{w^{(k)}_i}{\sum\limits^{n}_{j=1}w^{(k)}_j}exp\{2\hat{a_k} I\{y_i\neq \hat{C_k}(x_i)\}\}" class="ee_img tr_noresize" eeimg="1">
我们推出的公式为

<img src="https://www.zhihu.com/equation?tex=w_i^{k+1}=w_i^{k}e^{-y_i\hat{a_{k}}\hat{C_{k}}(x_i)}=w_i^{k}e^{-\hat{a_{k}}}e^{\hat{a_{k}}[1-y_i\hat{C_{k}}(x_i)]}" alt="w_i^{k+1}=w_i^{k}e^{-y_i\hat{a_{k}}\hat{C_{k}}(x_i)}=w_i^{k}e^{-\hat{a_{k}}}e^{\hat{a_{k}}[1-y_i\hat{C_{k}}(x_i)]}" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=而　1-y_i\hat{C_{k}}(x_i) ＝\left\{\begin{matrix}  0　y_i=\hat{C_k}(x_i) \\2　y_i\neq\hat{C_k}(x_i)
\end{matrix}\right.=2I\{y_i\neq \hat{C_k}(x_i)\}" alt="而　1-y_i\hat{C_{k}}(x_i) ＝\left\{\begin{matrix}  0　y_i=\hat{C_k}(x_i) \\2　y_i\neq\hat{C_k}(x_i)
\end{matrix}\right.=2I\{y_i\neq \hat{C_k}(x_i)\}" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=所以　 w_i^{k+1}=w_i^{k}e^{-\hat{a_{k}}}e^{2\hat{a_{k}}I\{y_i\neq \hat{C_k}(x_i)\}}" alt="所以　 w_i^{k+1}=w_i^{k}e^{-\hat{a_{k}}}e^{2\hat{a_{k}}I\{y_i\neq \hat{C_k}(x_i)\}}" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=\propto \frac{w^{(k)}_i}{\sum\limits^{n}_{j=1}w^{(k)}_j}exp\{2\hat{a_k} I\{y_i\neq \hat{C_k}(x_i)\}\}" alt="\propto \frac{w^{(k)}_i}{\sum\limits^{n}_{j=1}w^{(k)}_j}exp\{2\hat{a_k} I\{y_i\neq \hat{C_k}(x_i)\}\}" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=（\sum\limits^{n}_{j=1}w^{(k)}_j为归一化常数）" alt="（\sum\limits^{n}_{j=1}w^{(k)}_j为归一化常数）" class="ee_img tr_noresize" eeimg="1">


## 2.2 总结
 Adaboost 算法最终的的分类器是

<img src="https://www.zhihu.com/equation?tex=f(x_i)=\sum\limits^{T}_{t=1}a_tC_t(x_i)" alt="f(x_i)=\sum\limits^{T}_{t=1}a_tC_t(x_i)" class="ee_img tr_noresize" eeimg="1">
 所以实际上Adaboost想优化的问题是

<img src="https://www.zhihu.com/equation?tex=\sum\limits^{n}_{i=1}L(y_i,f(x_i))" alt="\sum\limits^{n}_{i=1}L(y_i,f(x_i))" class="ee_img tr_noresize" eeimg="1">
 
 但由于参数过多，直接优化容易导致过拟合问题，所以考虑加入惩罚项

<img src="https://www.zhihu.com/equation?tex=\sum\limits^{n}_{i=1}L(y_i,f(x_i))+\lambda_1 ||a||+\lambda_1 ||C||" alt="\sum\limits^{n}_{i=1}L(y_i,f(x_i))+\lambda_1 ||a||+\lambda_1 ||C||" class="ee_img tr_noresize" eeimg="1">
 此优化问题非常复杂，难以求解，所以我们采用上面这种**每次优化一个分类器**的优化算法。该算法不仅解决了复杂性的问题，而且达到的结果是和上式是一致的。



