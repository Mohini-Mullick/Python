def is_odd_even(num):
    if num % 2 == 0:
        return "Even"
    else:
        return "Odd"
    
num = int(input("Enter a number: "))

if num < 0:
    print("Invalid input. Please enter a positive integer.")
    exit()
    
print(f'The number {num} is {is_odd_even(num)}.')