import math
def sum_of_three_square_roots(numbers):
    sqrts = [math.sqrt(num) for num in numbers]
    sum = 0
    for num in sqrts:
        sum += num
    return sum

numbers = [float(input(f"Enter the first number: ")), float(input(f"Enter the second number: ")), float(input(f"Enter the third number: "))]
print(sum_of_three_square_roots(numbers))