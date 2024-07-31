def power(base, exponent):
    if exponent == 0:
        return 1

    return base * power(base, exponent-1)

base = float(input("Enter the base: "))
exponent = int(input("Enter the exponent: "))

result = power(base, exponent)

print(f"{base} raised to the power of {exponent} is: {result}")