# This is a sample Python script.
import math
import matplotlib.pyplot as plt
import numpy as np


def sin_to_taylor(x, pw):
    summ = 0
    for k in range(0, pw):
        summ += math.pow((-1), k) * (math.pow(x, 2 * k + 1) / math.factorial(2 * k + 1))
    return summ


def dots_for_plot():
    arr1 = []
    arr2 = []
    arr3 = []
    # for должен бегать не от 1 до 400, а от -2pi до +2pi, но чтобы было 400 шагов между ними
    for i in range(1, 400):
        arr1.append(sin_to_taylor(i, 1))
    for i in range(1, 400):
        arr2.append(sin_to_taylor(i, 3))
    for i in range(1, 400):
        arr3.append(sin_to_taylor(i, 7))
    plt.plot(arr1)
    plt.plot(arr2)
    plt.plot(arr3)
    plt.show()



if __name__ == '__main__':
    dots_for_plot()
