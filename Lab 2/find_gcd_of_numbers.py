def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

numbers = [int(input(f"Enter the first number: ")), int(input(f"Enter the second number: "))]
print(f"The GCD of {numbers[0]} and {numbers[1]} is {gcd(numbers[0], numbers[1])}")