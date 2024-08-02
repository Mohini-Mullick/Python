import cmath

def solve_quadratic(a, b, c):
    discriminant = (b**2) - (4*a*c)

    sol1 = (-b-cmath.sqrt(discriminant))/(2*a)
    sol2 = (-b+cmath.sqrt(discriminant))/(2*a)

    return sol1, sol2


a = 1
b = -3
c = 2

sol1, sol2 = solve_quadratic(a, b, c)

print(f'The solutions of the quadratic equation {a}x² + {b}x + {c} = 0 are {sol1} and {sol2}')