
def nedo_integral(step,amp):
    x = 0
    summ = 0
    while x <= 1:
        summ += step * amp
        x += step
    return summ


if __name__ == '__main__':
    print(nedo_integral(0.01,2))
