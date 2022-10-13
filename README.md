# FCM_RBF_Control
> 自适应FCM RBF预测，MPC控制模型。
### RBF
```
RBF（径向基神经网络）部分-复现来源于
 (孙剑,蒙西,乔俊飞.城市固废焚烧过程烟气含氧量自适应预测控制[J/OL].自动化学报:p5.)
```
> 计算公式如下

![avatar](/img/img.png)
![avatar](/img/img_1.png)

### FCM 
> 自适应FCM聚类算法迭代计算出训练样本的聚类数目和聚类中心作为RBF神经网络的隐含层神经元个数和初始中心.
```
来源于:成熟的框架fuzzy-c-mean
```
