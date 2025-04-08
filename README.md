# CV PJ1
本次实验的代码由以下Python文件构成：
- download.py: 下载CIFAR10数据集
- visualize.py: 可视化图片和权重参数
- preprocess.py: 数据预处理
- mlp.py: MLP类定义
- optimizer.py: 优化器
- train.py: 训练模型
- test.py: 测试模型
- hyperparam optim.py: 超参数查找

训练模型只需在train.py中修改必要的参数并运行即可，测试模型需移步至test.py运行。train.py会
将训练得到的模型以二进制文件的形式保存在文件夹best model下，默认命名为best_model.pickle，
运行test.py时会从best_model.pickle中加载模型并完成测试。

Google Drive链接包含的文件夹saved model下保存了本
报告中的三个模型，分别是小数据集上的过拟合训练模型best_modelSM.pickle，查找超参数
后的训练模型best_modelHP.pickle，以及加入BN层后的模型best_modelBN.pickle。注意，加
载前两个模型用到MLP的类函数load_model_without_BN，而加载BN模型用到MLP的类函数load_model。

CIFAR10数据集在文件夹data/cifar10下，其中
- test_batch为测试集，包含10000个样本
- data_batch 1∼5为训练集，每批包含10000个样本，共50000个样本

Github Repository链接：
- https://github.com/Dear-Cry/QspjGqlh.git
  
Google Drive链接：
- https://drive.google.com/drive/folders/1ycJgqkhqbJm7vHAVdqFr2_IwjticjD25?usp=sharing
