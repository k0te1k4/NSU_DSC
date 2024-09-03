import math
import matplotlib.pyplot as plt


def sin_to_taylor(x, pw):
    summ = 0
    for k in range(0, pw):
        summ += math.pow((-1), k) * (math.pow(x, 2 * k + 1) / math.factorial(2 * k + 1))
    return summ

def dots_for_plot(count,pw):
    arr = []
    piece = (4 * 3.14)/count # (2 * 3.14 - (-2 * 3.14))/count
    for i in range(1, count):
        arr.append(sin_to_taylor(-2 * 3.14 + piece * i, pw))
    return arr

def show_plot(count):
    plt.plot(dots_for_plot(count,1))
    plt.plot(dots_for_plot(count,3))
    plt.plot(dots_for_plot(count,7))
    plt.show()



if __name__ == '__main__':
    show_plot(400)

