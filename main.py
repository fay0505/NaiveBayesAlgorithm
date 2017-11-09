from LoadData import loaddata
from SeparateData import separatedata
from Splitbyclass import splitbyclass
from Predict import classify
import LearnfromTrainset


if __name__ == '__main__':


    #给出数据路径，并将数据存入列表
    Path = 'pima-indians-diabetes.data'
    Dataset = loaddata(Path)

    #划分数据集
    Ratio = float(input("请输入训练集所占比例（大小在0-1之间）："))
    Trainset, Predictset = separatedata(Ratio, Dataset )

    #把训练集按照类别划分
    Class0 , Class1 = splitbyclass(Trainset)

    #计算先验概率
    Prior0 , Prior1 = LearnfromTrainset.prior_p(Class0, Class1)

    #计算每一个类别中每种属性的均值与方差
    M0 , V0 = LearnfromTrainset.meanAndvariance(Class0)
    M1 , V1 = LearnfromTrainset.meanAndvariance(Class1)

    #利用学习到的知识进行预测
    Accuracy = classify(Predictset, M0, V0, M1, V1, Prior0, Prior1)
    print("Accuracy:", Accuracy)


