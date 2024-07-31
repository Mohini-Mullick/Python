def print_multiplication_table(n, upto=10):
    print(f"Multiplication Table of {n}:")
    for i in range(1, upto+1):
        print(f"{n} x {i} = {n * i}")

try:
    number = int(input("Enter a number: "))
    if number < 1:
        raise ValueError("Input must be a positive integer.")
        
    print_multiplication_table(number)
except ValueError as e:
    print(e)