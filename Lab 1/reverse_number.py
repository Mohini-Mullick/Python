number = int(input("Enter a number: "))
final_number = ''

reversed_number = 0
while number > 0:

    remainder = number % 10
    
    reversed_number = reversed_number * 10 + remainder
    
    number = number // 10
    
    final_number = final_number + str(remainder)
    
print("Reversed number:", final_number)