from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random, time
from varname import nameof
PI = np.pi
def draw_rastrigin_function():
    X = np.linspace(-5.12, 5.12, 100)
    Y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(X, Y)

    Z = (X ** 2 - 10 * np.cos(2 * PI * X)) + \
        (Y ** 2 - 10 * np.cos(2 * PI * Y)) + 20
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap=cm.plasma, linewidth=0,
                    antialiased=True)
    plt.show()
def rastrigin_function(X, Y):
    Z = (X ** 2 - 10 * np.cos(2 * PI * X)) + \
        (Y ** 2 - 10 * np.cos(2 * PI * Y)) + 20
    return Z

def draw_cross_in_tray_function():
    X = np.linspace(-10, 10, 100)
    Y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(X, Y)

    Z = -0.0001 * ((np.abs(np.sin(X) * np.sin(Y) * np.exp(np.abs(100 - (np.sqrt((X ** 2) + (Y ** 2)) / PI)))) + 1) ** 0.1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap=cm.plasma, linewidth=0,
                    antialiased=True)
    plt.show()
def cross_in_tray_function(X, Y):
    Z = -0.0001 * ((np.abs(np.sin(X) * np.sin(Y) * np.exp(np.abs(100 - (np.sqrt((X ** 2) + (Y ** 2)) / PI)))) + 1) ** 0.1)
    return Z

def draw_three_hump_camel_function():
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)

    Z = 2 * X ** 2 - 1.05 * X ** 4 + (X ** 6) / 6 + X * Y + Y ** 2
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap=cm.plasma, linewidth=0,
                    antialiased=True)
    plt.show()
def three_hump_camel_function(X,Y):
    Z = 2 * X ** 2 - 1.05 * X ** 4 + (X ** 6) / 6 + X * Y + Y ** 2
    return Z

def evaluate_function(function, min_domain, max_domain, perfect_value):
    result_vector = [0, 0, 0, 0]
    result_vector[3] = perfect_value
    result = 1000
    for i in range(1000000):
        X = random.uniform(min_domain, max_domain)
        Y = random.uniform(min_domain, max_domain)
        if function(X, Y) < result:
            result = function(X,Y)
            result_vector[0] = X
            result_vector[1] = Y
            result_vector[2] = function(X,Y)
    return result_vector

def rate_evaluate(list):
    result = list[3] + 1
    result -= list[2]
    return result

#sdraw_rastrigin_function()
#draw_cross_in_tray_function()
#draw_three_hump_camel_function()
t_start = time.process_time()
a = evaluate_function(rastrigin_function, -5.12, 5.12, 0)
print(a)
t_elapsed = time.process_time() - t_start
print(t_elapsed)
print("Rozwiązanie jest w " + str(rate_evaluate(a)*100) + "% idealne")

t_start = time.process_time()
b = evaluate_function(cross_in_tray_function, -10, 10, -2.06261)
print(b)
t_elapsed = time.process_time() - t_start
print(t_elapsed)
print("Rozwiązanie jest w " + str(rate_evaluate(b)*100) + "% idealne")

t_start = time.process_time()
c = evaluate_function(three_hump_camel_function, -5, 5, 0)
print(c)
t_elapsed = time.process_time() - t_start
print(t_elapsed)
print("Rozwiązanie jest w " + str(rate_evaluate(c)*100) + "% idealne")
