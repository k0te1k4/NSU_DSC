import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.bipartite import color
from roboflow.util.prediction import plot_annotation
from scipy import integrate
from scipy import signal

def an(t0, x, big_t, n, w):

    def fun(t):
        return x(t,big_t) * np.cos(n * w * t)

    return (2 / big_t) * integrate.quad(lambda t: func(t, big_t) * np.cos(n * w * t), t0, t0 + big_t)[0]

def bn(t0, x,big_t, n, w):

    def fun(t):
        return x(t,big_t) * np.sin(n * w * t)

    return (2 / big_t) * integrate.quad(lambda t: func(t, big_t) * np.sin(n * w * t), t0, t0 + big_t)[0]


def approx_func(t0, x ,t, big_t, w, big_n):
    a0 = an(t0, x,big_t, 0, w)
    summ = 0
    summ = a0 / 2 + summ
    for n in range(1, big_n):
        summ += an(t0, x,big_t, n, w) * np.cos(n * w * t) + bn(t0, x,big_t, n, w) * np.sin(n * w * t)
    return summ



def func(t, big_t):
    A = 2
    if  t % big_t > big_t / 2:
        return A
    return -A



if __name__ == '__main__':
    pi = np.pi
    big_n = 10
    big_t = 2
    w = 2 * pi / big_t
    t0 = 0


    plot_t = np.arange(-4,4,0.01)



    plot_fun = list(map(lambda a: func(a,2),plot_t))
    plot_approx = list(map(lambda a: approx_func(0, func , a, big_t, w, 10),plot_t))
    plot_err = np.subtract(np.array(plot_approx),np.array(plot_fun))
    plt.plot(plot_t,plot_fun)
    plt.plot(plot_t,plot_approx)
    plt.show()
    plt.plot(plot_t,plot_err)
    plt.show()
