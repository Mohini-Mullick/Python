import math

def is_krishnamurthy_number(n):
    # Convert number to string for easier manipulation
    digits = str(n)
    
    sum_of_factorials = sum(math.factorial(int(digit)) for digit in digits)
    
    return sum_of_factorials == n

number = int(input("Enter a number: "))

if is_krishnamurthy_number(number):
    print(number, "is a Krishnamurthy number.")
else:
    print(number, "is not a Krishnamurthy number.")