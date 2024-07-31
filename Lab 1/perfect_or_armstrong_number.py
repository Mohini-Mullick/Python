num = int(input("Enter a number: "))
def is_perfect(n):
    sum = 0
    for i in range(1, n):
        if(n % i == 0):
            sum = sum + i
    if (sum == n):
        return True
    else:
        return False
                
def is_armstrong(n):
    sum = 0
    temp = num
    while temp > 0:
       digit = temp % 10
       sum += digit ** 3
       temp //= 10


    if num == sum:
       return True
    else:
       return False

if is_perfect(num):
    print(f"{num} is a perfect number.")
else:
    print(f"{num} is not a perfect number.")
    
if is_armstrong(num):
    print(f"{num} is an Armstrong number.")
else:
    print(f"{num} is not an Armstrong number.")