num1, num2, num3 = float(input("Enter the first number: ")), float(input("Enter the second number: ")), float(input("Enter the third number: "))

if num1 >= num2 and num1 >= num3:
    max_num = num1
elif num2 >= num1 and num2 >= num3:
    max_num = num2
else:
    max_num = num3
    
print("The maximum number is:", max_num)