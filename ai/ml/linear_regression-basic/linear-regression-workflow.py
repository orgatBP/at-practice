import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


###�ռ�׼������
def get_dataset(csv_file):
    data = pd.read_csv(csv_file)
    # print(data.head()) #��ʾǰ5������
    return data


###�۲����ݷֲ�
def figure_dataset(data):
    # ��ͬһͼ���й۲��������ݵķֲ����
    plt.plot(data["TV"], "ro", label="TV")
    plt.plot(data["Radio"], "g^", label="Radio")
    plt.plot(data["Newspaper"], "b*", label="Newspaper")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    # ͨ������ͼ�۲�����ݷֲ������
    plt.figure(figsize=(9, 12))
    plt.subplot(311)
    plt.plot(data["TV"], "ro")
    plt.title("TV")
    plt.grid()

    plt.subplot(312)
    plt.plot(data["Radio"], "g^")
    plt.title("Radio")
    plt.grid()

    plt.subplot(313)
    plt.plot(data["Newspaper"], "b*")
    plt.title("Newspaper")
    plt.grid()

    plt.tight_layout()
    plt.show()


###���������ͱ�ǩ
def get_features(data):
    feature_cols = ["TV", "Radio", "Newspaper"]

    # use the list to select a subset of original DataFrame
    x = data[feature_cols]

    # check the type and shape of X
    print(type(x))
    print(x.shape)

    # select a series from the DataFrame
    y = data["Sales"]

    return x, y


###����ѵ���Ͳ��Լ���
def get_train_dataset(x, y):
    # Ĭ���з���75%��ѵ������25%����֤��
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)

    return x_train, x_test, y_train, y_test


###sklearn�����Իع�
def fit(x_train, y_train, x_test, y_test):
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    # print(model)
    # print(linreg.intercept_)
    # print(linreg.coef_)

    y_pred = linreg.predict(x_test)
    # print(y_pred)
    # print(type(y_pred),type(y_test))
    # print(len(y_pred),len(y_test))
    # print(y_pred.shape,y_test.shape)

    return y_pred, y_test


###�ع����������
# ƽ���������(Mean Absolute Error,MAE)
# �������(Mean Squared Error,MSE)
# ���������(Root Mean Squared Error,RMSE)
def evaluation(y_pred, y_test):
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
        print("RMSE by hand", np.sqrt(sum_mean / len(y_pred)))

    # ��ͼֱ�۸���
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, "b", label="predict")
    plt.plot(range(len(y_test)), y_test, "r", label="test")
    plt.legend(loc="upper right")  # ��ʾͼ�еı�ǩ
    plt.xlabel("the number of sales")
    plt.ylabel("value of sales")
    plt.show()


###�������
###���ڹ�ʽ�ͽ��,���Ƿ��б�Ҫ

if __name__ == "__main__":
    csv_file = r"D:\Advertising.csv"
    data = get_dataset(csv_file)
    # figure_dataset(data)
    x, y = get_features(data)
    x_train, x_test, y_train, y_test = get_train_dataset(x, y)

    y_pred, y_test = fit(x_train, y_train, x_test, y_test)
    print("============================================")

    print(y_pred)

    print("============================================")
    print(y_test)

    evaluation(y_pred, y_test)
