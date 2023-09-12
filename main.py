from collections import Counter
from math import log
import operator
import numpy as np


def create_train_dataset():  # 创造示例数据
    data_set = np.array([
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '稍缩', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍缩', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍缩', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
        ['乌黑', '稍缩', '沉闷', '稍糊', '稍凹', '硬滑', '好瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '硬滑', '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '软粘', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        ['青绿', '稍缩', '浊响', '稍糊', '凹陷', '软粘', '坏瓜'],
        ['浅白', '稍缩', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['乌黑', '稍缩', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ])

    attributes = np.array(['色泽', '根蒂', '敲声', '纹理', '脐部', '触感'])  # 6个特征
    return data_set, attributes


def create_test_dataset():  # 创造示例数据
    data_set = np.array([
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑']
    ])
    y_labels = np.array([1.0, 0.0])

    return data_set, y_labels


class DecisionTree:
    def __init__(self, max_depth: int = 10, criterion: str = "gini"):
        self.max_depth = max_depth
        self.criterion = criterion
        self.data_set = None
        self.attributes = None
        self.tree = None

    def calc_entropy(self, data_set):  # 计算数据的熵(entropy)
        sample_num = len(data_set)  # 数据条数
        label_counts = Counter()

        for feat_vec in data_set:
            current_label = feat_vec[-1]  # 每行数据的最后一个字（类别）
            label_counts[current_label] += 1  # 统计有多少个类以及每个类的数量

        entropy_value = 0

        for key in label_counts:
            prob = float(label_counts[key]) / sample_num  # 计算单个类的熵值

            if self.criterion == "gini":
                loss_item = 1 - prob
            elif self.criterion == "id3":
                loss_item = -log(prob, 2)
            else:
                raise NotImplementedError

            entropy_value += prob * loss_item  # 累加每个类的熵值

        return entropy_value

    @staticmethod
    def split_dataSet(data_set, axis, value):  # 按某个特征分类后的数据
        retDataSet = []
        for featVec in data_set:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    @staticmethod
    def majorityCnt(classList):  # 按分类后类别数量排序，比如：最后分类为2好瓜1坏瓜，则判定为好瓜；
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def choose_best_feature_to_split(self, data_set):  # 选择最优的分类特征
        num_features = len(data_set[0]) - 1
        base_entropy = self.calc_entropy(data_set)  # 原始的熵

        best_info_gain = 0
        best_feature = -1

        for i in range(num_features):
            feat_list = [example[i] for example in data_set]
            unique_values = set(feat_list)
            new_entropy = 0

            for value in unique_values:
                sub_data_set = self.split_dataSet(data_set, i, value)
                prob = len(sub_data_set) / float(len(data_set))

                # 计算label熵值
                new_entropy += prob * self.calc_entropy(sub_data_set)  # 按特征分类后的熵

            info_gain = base_entropy - new_entropy  # 原始熵与按特征分类后的熵的差值
            if info_gain > best_info_gain:  # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
                best_info_gain = info_gain
                best_feature = i

        return best_feature

    def create_tree(self, data_set, y_attributes, depth=0):
        classList = [example[-1] for example in data_set]  # 类别：好瓜或坏瓜

        if classList.count(classList[0]) == len(classList):
            return classList[0]

        if len(data_set[0]) == 1 or depth == self.max_depth:
            return self.majorityCnt(classList)

        best_feat = self.choose_best_feature_to_split(data_set)  # 选择最优特征
        best_feat_label = y_attributes[best_feat]

        tree = {best_feat_label: {}}  # 分类结果以字典形式保存
        del (y_attributes[best_feat])

        feat_values = [example[best_feat] for example in data_set]
        unique_values = set(feat_values)

        for value in unique_values:
            sub_labels = y_attributes[:]
            tree[best_feat_label][value] = self.create_tree(
                self.split_dataSet(data_set, best_feat, value),
                sub_labels,
                depth + 1
            )

        self.tree = tree

    def _predict_sample(self, x, attribute_is, tree):
        if not isinstance(tree, dict):
            return 1 if tree == "好瓜" else 0

        # 访问 属性 相应的 属性值 例如: 纹理->模糊
        attribute = list(tree.keys())[0]
        attribute_idx = attribute_is.index(attribute)
        attribute_value = x[attribute_idx]

        return self._predict_sample(x, attribute_is, tree[attribute][attribute_value])

    def predict(self, X, attrs):
        predictions = []
        for x in X:
            predictions.append(self._predict_sample(x, attrs, self.tree))
        return np.array(predictions)


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, criterion=None, random_state=None,
                 samples_percent=0.75, features_percent=0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self.samples_percent = samples_percent
        self.features_percent = features_percent
        self.trees = list()
        self.bound_marks = {}

    def fit(self, X, y):
        np.random.seed(self.random_state)

        max_samples = X.shape[0]
        max_features = X.shape[1] - 1

        # 标记每个样本是否被访问过
        self.bound_marks = {sample_idx: False for sample_idx in range(max_samples)}
        for _ in range(self.n_estimators):
            DT = DecisionTree(max_depth=self.max_depth, criterion=self.criterion)

            # 选取样本行
            subset_indices = np.random.choice(
                a=range(max_samples),
                size=int(self.samples_percent * max_samples),
                replace=True
            )

            # 包外估计样本
            for subset_ind in subset_indices:
                if self.bound_marks[subset_ind]:
                    continue
                self.bound_marks[subset_ind] = True

            # 选取特征列
            subset_features = np.random.choice(
                a=range(max_features),
                size=int(self.features_percent * max_features),
                replace=False
            )
            subset_features = np.append(subset_features, [-1], axis=0)

            subset_X = X[subset_indices][:, subset_features].tolist()
            subset_y = y[subset_features].tolist()

            DT.create_tree(subset_X, subset_y)
            self.trees.append(DT)

    def out_bound_estimates(self):
        """包外估计"""
        evaluations = list()
        for key, value in self.bound_marks.items():
            if not value: evaluations.append(key)

        return evaluations

    def predict(self, X, headers):
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for i, tree in enumerate(self.trees):
            pred = tree.predict(X, headers)
            predictions[:, i] = pred
        return [Counter(pred).most_common(1)[0][0] for pred in predictions]


def calculate_accuracy(y_pred, y_fact):
    corrects = np.equal(y_pred, y_fact)
    return np.mean(corrects)


if __name__ == '__main__':
    # 训练
    data_set, attributes = create_train_dataset()
    RF = RandomForest(n_estimators=10, max_depth=10, criterion="gini", random_state=42)
    RF.fit(data_set, attributes)

    # 预测
    eval_data_set, y_labels = create_test_dataset()
    y_preds = RF.predict(eval_data_set, attributes.tolist())
    print("预测结果:", y_preds)
    acc = calculate_accuracy(y_preds, y_labels)
    print("准确率为: {}%".format(acc*100))
