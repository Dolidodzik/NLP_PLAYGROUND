'''
this file saves .csv of argument-value (x,y) pairs for given cubic expression, with some noise. 
'''

import pandas as pd
import numpy as np

a = 1
b = -2
c = -8
d = 10
maximum_noise = 0.5
pairs_count = 500
x_range = 100

filename = f"REGRESSION_OF_FUNCTION_a={a}_b={b}_c={c}_d={d}.csv"

x_values = np.random.uniform(100, -100, [pairs_count])
y_values = a * x_values**3 + b * x_values**2 + c * x_values + d

print("\nvalues beofre appying noise")
print("x values: ", x_values)
print("y values: ", y_values)

x_noise = np.random.uniform(maximum_noise, -maximum_noise, [pairs_count])
y_noise = np.random.uniform(maximum_noise, -maximum_noise, [pairs_count])

x_values += x_noise
y_noise += y_noise

print("\nvalues after appying noise: ")
print("x values: ", x_values)
print("y values: ", y_values)

df = pd.DataFrame({'x': x_values, 'y': y_values})
df.to_csv(filename, index=False)