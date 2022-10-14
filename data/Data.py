# -*- coding: utf-8 -*-
# @Time : 2022/9/19 9:57
# @Author : Wang Hai
# @Email : nicewanghai@163.com
# @Code Specification : PEP8
# @File : Data.py
# @Project : FCM_RBF_NN
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def get_data():
    # 取数据 数据涉及保密，所以没有上传，其实就是一个csv文件，下面中文就是列名，行数大于12000.
    data_path = r"./data/Boiler_2022_08_10-15.csv"
    read_data = pd.read_csv(data_path, index_col=0).iloc[:12000, 0:18]
    all_data = pd.DataFrame([])
    all_data["总给煤"] = read_data['1#给煤机瞬时流量输出信号'] + read_data['2#给煤机瞬时流量输出信号'] + read_data['3#给煤机瞬时流量输出信号'] + read_data['4#给煤机瞬时流量输出信号']
    all_data['一次风总风量'] = read_data['一次风总风量']
    all_data['二次风总流量'] = read_data['二次风总流量']
    all_data['平均氧量'] = (read_data['高温省煤器进口O2浓度（左）'] + read_data['高温省煤器进口O2浓度（右）']) / 2
    # 训练集
    train_datas = all_data[:10000]
    df_x = train_datas.copy()[:-10].reset_index(drop=True)
    df_y = train_datas.copy()[10:]['平均氧量'].reset_index(drop=True)
    train_data, eval_data, train_labels, eval_labels = train_test_split(df_x, df_y, test_size=0.2)
    x_t = np.array(train_data).round(2)
    y_t = np.array(train_labels).round(2).reshape(-1, 1)
    x_ = np.array(eval_data).round(2)
    y_ = np.array(eval_labels).round(2).reshape(-1, 1)
    # 归一化
    max_abs_scaler = preprocessing.MaxAbsScaler()
    x_t_minmax = max_abs_scaler.fit_transform(x_t)
    x_minmax = max_abs_scaler.fit_transform(x_)
    # 测试集
    test_datas = all_data[10000:]
    te_x = test_datas.copy()[:-10].reset_index(drop=True)
    te_y = test_datas.copy()[10:]['平均氧量'].reset_index(drop=True)
    te_x = np.array(te_x).round(2)
    te_y = np.array(te_y).round(2).reshape(-1, 1)
    # 归一化
    te_x = max_abs_scaler.fit_transform(te_x)
    return x_t_minmax, y_t, x_minmax, y_, te_x, te_y
