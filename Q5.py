import numpy as np
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def solve(x_vector, y_vector):
    new_x = x_vector
    # new_y = log (y / x)
    new_y = np.divide(y_vector, x_vector)
    new_y = [math.log(y) for y in new_y]

    curve = np.polyfit(new_x, new_y, 1)  # Ax + B

    b = -curve[0]  # b = -A => b = -A
    a = math.exp(curve[1])  # ln a = B => e^B = a

    # y = a * x * math.exp(-b*x)
    y_2 = lambda x: a * x * math.exp(-b * x)

    return y_2


x_measure = np.array([0.25, 0.5, 1, 2, 3, 4, 5])
y_measure1 = np.array([0.9, 1.2, 0.5, 0.15, 0.033, 0.005, 0.0001])
y_measure2 = np.array([0.9, 1.2, 0.5, 0.15, 0.033, 0.005, 0.001])

line1 = solve(x_measure, y_measure1)
line2 = solve(x_measure, y_measure2)

line1_y = np.array([line1(x) for x in x_measure])
line2_y = np.array([line2(x) for x in x_measure])

print('part 1')
print([list(a) for a in zip(x_measure, line1_y)])

print('part 2')
print([list(a) for a in zip(x_measure, line2_y)])


f, axarr = plt.subplots(2, 1)

axarr[0].plot(x_measure, y_measure1, color='red', label="data")
axarr[1].plot(x_measure, y_measure2, color='red', label="data")

axarr[0].set_title("Part 1")
axarr[0].plot(x_measure, line1_y, color='blue', label="curve fitting")

axarr[1].set_title("Part 2")
axarr[1].plot(x_measure, line2_y, color='blue', label="curve fitting")


plt.legend()
plt.show()




