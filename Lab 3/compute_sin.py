import math

def compute_sin(x, n):
    sin_value = 0
    for i in range(n):
        sign = (-1) ** i
        factorial = math.factorial(2 * i + 1)
        term = sign * (x ** (2 * i + 1)) / factorial
        sin_value += term
    return sin_value

x = float(input("Enter the value of x: "))
n = int(input("Enter the number of terms: "))

sin_x = compute_sin(x, n)
print(f"The sin of {x} using {n} terms is: {sin_x}")