# -*- coding: utf-8 -*-
# @Time : 2022/9/16 11:22
# @Author : Wang Hai
# @Email : nicewanghai@163.com
# @Code Specification : PEP8
# @File : FCM_RBF.py
# @Project : FCM_RBF_NN
import logging
import numpy as np
from fcmeans import FCM
from data.Data import get_data
from scipy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RBF:
    def __init__(self, x_, y_):
        # 初始化训练数据用于在FCM上得出初始中心和神经元个数
        self.x_ = x_
        self.y_ = y_
        # 初始化FCM模型
        fcm = FCM()
        # 训练
        fcm.fit(x_)
        # 初始中心·(FCM计算得到)
        self.c = fcm.centers
        # 神经元的个数 (FCM计算得到)
        self.hc = len(fcm.centers)
        # 初始化中心宽度
        self.h = np.random.random(self.hc)
        # 初始化神经元的权重
        self.w = np.random.random(self.hc)
        # 初始化学习率
        self.lr = 0.01
        # 迭代次数
        self.iters = 7
        # 初始化日志
        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                            level=logging.INFO,
                            filename='./log/参数.log',
                            filemode='a')

    @staticmethod
    # 高斯函数也就是径向基函数
    def _gaussian_function(x_i, c_j, h_j):
        """
        x: 输入特征
        c_j:cj 和 δj 分别是第 j 个神经元的中心向量和宽度
        """
        result = np.exp(-norm(x_i - c_j) ** 2 / (2 * h_j ** 2))
        return result

    # 每个神经元计算后求和得出的预测结果
    def y_out(self, x_i):
        out = 0
        for j in range(self.hc):
            out += self.w[j] * self._gaussian_function(x_i, self.c[j], self.h[j])
        return out

    # 误差函数
    @staticmethod
    def _error(y_p, y_t):
        return (y_p - y_t) ** 2 / 2

    # 参数更新
    def update_w(self, y_p, y_t, x_i):
        for j in range(self.hc):
            self.c[j] = self.c[j] - self.lr * ((y_p - y_t) * (self.w[j] / self.h[j] ** 2)) * \
                        self._gaussian_function(x_i, self.c[j], self.h[j]) * (x_i - self.c[j])
            self.h[j] = self.h[j] - self.lr * ((y_p - y_t) * (self.w[j] / self.h[j] ** 3)) * self._gaussian_function(x_i, self.c[j], self.h[j]) * (norm(x_i - self.c[j]) ** 2)
            self.w[j] = self.w[j] - self.lr * (y_p - y_t) * self._gaussian_function(x_i, self.c[j], self.h[j])
        return self.c, self.h, self.w

    # 训练
    def train(self):
        error_list = []
        for i in range(self.iters):
            tmp = ()
            for i_ in range(len(self.x_)):
                y_p = self.y_out(self.x_[i_])
                y_t = self.y_[i_]
                tmp = self.c, self.h, self.w
                self.update_w(y_p, y_t, self.x_[i_])
                error_list.append(self._error(y_p, y_t))
            if len(error_list) >= 2:
                # 如果误差没有减小，参数不更新
                if error_list[-1] > min(error_list[:-1]):
                    self.c, self.h, self.w = tmp
            print("第{}次迭代后，最小误差为:{}".format(i+1, min(error_list)))

    # 预测
    def predict(self, x_test):
        p = []
        for x_v in x_test:
            p.append(self.y_out(x_v))
        return p

    @staticmethod
    def predict_metrics(eval_labels, predict_s):
        mae = mean_absolute_error(eval_labels, predict_s)
        r_mse = np.sqrt(mean_squared_error(eval_labels, predict_s))
        r2 = r2_score(eval_labels, predict_s)
        out_str_01 = "平均绝对误差:{},均方根误差:{}, r2:{}".format(mae, r_mse, r2)
        logging.info(out_str_01)
        logging.info("-----------------------------------------------------------------------")
        print(out_str_01)


if __name__ == "__main__":
    # 获取数据，训练集测试集数据
    x_train, y_train, x_eval, y_eval, te_x, te_y = get_data()
    rbf = RBF(x_train, y_train)
    rbf.train()
    y_predicts = rbf.predict(x_eval)
    y_test = rbf.predict(te_x)
    label = [i for i in range(len(y_eval))]
    out_str = "最终神经元参数:\n中心向量:{}\n宽度:{}\n权重:{}".format(rbf.c, rbf.h, rbf.w)
    logging.info(out_str)
    # print(out_str)
    rbf.predict_metrics(y_eval, y_predicts)
    rbf.predict_metrics(te_y, y_test)
    plt.figure(figsize=(10, 3))
    plt.plot(label[-50:], te_y[-50:], label="true")
    plt.plot(label[-50:], y_test[-50:], label="predict")
    plt.legend()
    plt.show()

