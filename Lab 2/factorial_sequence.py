import math

def print_factorial_series(n):
    for i in range(1, n+1):
        factorial = math.factorial(i)
        print(factorial)
        
num=6
print_factorial_series(num)