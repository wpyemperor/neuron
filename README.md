# neuron
本代码提取swc文件特征后，利用机器学习方法，实现对神经元局部结构的生物学类型判别

## 主要依赖
pyzem
https://github.com/janelia-flyem/pyzem　
sklearn

## 代码运行
生物学类型判别
运行ne_code/feature/debug_cfw.py。可以提取测试swc文件的特征。
（修改830以及974行文件路径）
运行ne_code/network/scmcv下代码，可分别输出随机森林（skrf_scvcm），决策树（skdt_scvcm），svm（svm_scvcm.py）的训练结果以及混淆矩阵

错误预测
运行ne_code/feature/error.py。可以提取测试swc文件的特征。（同理修改文件路径）
运行ne_code/network/scmcv下代码，可分别输出随机森林（skrf_scvcm），决策树（skdt_scvcm），svm（svm_scvcm.py）的训练结果以及混淆矩阵

## 其余代码说明
