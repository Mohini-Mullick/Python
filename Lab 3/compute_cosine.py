import math

def compute_cosine(x, n):
    cosine_value = 0
    for i in range(n):
        sign = (-1) ** i
        factorial = math.factorial(2 * i)
        term = sign * (x ** (2 * i)) / factorial
        cosine_value += term
    return cosine_value

x = float(input("Enter the value of x: "))
n = int(input("Enter the number of terms: "))

cosine_x = compute_cosine(x, n)
print(f"The cosine of {x} using {n} terms is: {cosine_x}")