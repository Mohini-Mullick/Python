import math

def solve_quadratic_equation(a, b, c):
    d = b**2 - 4*a*c
    root1 = (-b + math.sqrt(d)) / (2*a)
    root2 = (-b - math.sqrt(d)) / (2*a)
    
    return root1, root2

numbers = [float(input(f"Enter the coefficient of a: ")), float(input(f"Enter the coefficient of b: ")), float(input(f"Enter the coefficient of c: "))]
solutions = solve_quadratic_equation(numbers[0], numbers[1], numbers[2])

print(f"The solutions are: {solutions[0]} and {solutions[1]}")