# Day4 手推6条

## 1. 交叉熵损失 vs Brier损失
交叉熵损失：$L=-\sum y_i \log\hat y_i$，侧重概率分布的相对差异；对置信错判惩罚更大，分类任务常用。
Brier损失（评分损失）：$L=\sum (\hat y_i-y_i)^2$，是概率的均方误差，衡量绝对偏差；损失有界，对异常样本更稳健，可解释性好。

## 2. L1正则 vs L2正则
$L_1: \Omega(w)=\lambda\sum|w_i|$，产生稀疏解，可做特征选择；在0点不可导。
$L_2: \Omega(w)=\lambda\sum w_i^2$，权重衰减，把参数往0附近压缩，参数值平滑，不会得到严格稀疏。

## 3. 偏差-方差分解
总误差 = 偏差$^2$ + 方差 + 噪声。
偏差：模型本身拟合能力不足，欠拟合；方差：模型对训练集噪声敏感，换数据集预测波动大，过拟合；噪声是数据本身固有不可约误差。

## 4. 模型AUC差异检验要点
两组AUC置信区间重叠**不能判定差异不显著**。置信区间是各自单点估计的不确定性；检验两个AUC是否有差异，要检验**AUC差值的分布**。
推荐：成对Bootstrap构造AUC差值CI，或者DeLong检验。
AUC衡量全阈值区分能力；McNemar检验是固定决策切点下，样本分类对错的配对检验。
结论：两个模型AUC无显著差异，但是在选定阈值附近，样本判定结果差异大，所以McNemar显著。

## 5. BN的train/eval行为
train模式：使用batch内均值、方差；同时Dropout开启，两次前向结果不同。
eval模式：使用训练阶段累积的running统计量；Dropout关闭，相同输入多次前向输出完全一致。

## 6. XGBoost二阶泰勒展开
符号定义：
$\hat y_i^{(t-1)}$：前$t-1$棵树累积预测值。
$g_i=\partial_{\hat y_i^{(t-1)}} l\left(y_i,\hat y_i^{(t-1)}\right)$，损失一阶梯度；
$h_i=\partial^2_{\hat y_i^{(t-1)}} l\left(y_i,\hat y_i^{(t-1)}\right)$，损失二阶梯度（海塞项）。
$f_t(x_i)$：第$t$棵待学习树；正则项$\Omega(f)=\gamma T+\frac12\lambda\sum_{j=1}^T w_j^2$，$T$叶子数，$w_j$叶子权重。

$$
L^{(t)}=\sum_{i=1}^n l\left(y_i,\hat y_i^{(t-1)}+f_t(x_i)\right)+\Omega(f_t)
$$
二阶泰勒展开：
$$
l(\cdot)\approx l(y_i,\hat y_i^{(t-1)})+g_i f_t(x_i)+\frac12 h_i f_t^2(x_i)
$$
舍弃常数项：
$$
L^{(t)} \approx \sum_i \big[g_i f_t(x_i)+\tfrac12 h_i f_t^2(x_i)\big]+\Omega(f_t)
$$
设叶子$j$样本集合$I_j=\{i|x_i\in叶子j\}$，$G_j=\sum_{i\in I_j}g_i,\ H_j=\sum_{i\in I_j}h_i$。
$$
L=\sum_{j=1}^T\left[G_j w_j+\frac12(H_j+\lambda)w_j^2\right]+\gamma T
$$
对$w_j$求导并置0：
$$
w_j^* = -\frac{G_j}{H_j+\lambda}
$$
把最优$w_j^*$代回，最优目标：
$$
L^* = -\frac12 \sum_j \frac{G_j^2}{H_j+\lambda}+\gamma T
$$
> 该式是分裂Gain的来源，Gain = 分裂前后最优目标之差。

灵魂问答：为什么用$g,h$而不用残差？
- 平方损失下 $g_i=\hat y^{(t-1)}-y_i$，就是残差，**g/h是残差的推广**。
- 推广的意义：任何二阶可导损失（logistic、pairwise）都能套用同一套求解框架，损失函数可插拔。
- 二阶信息 = 对损失做局部二次逼近，比GBDT只用一阶的线性逼近更精准。
