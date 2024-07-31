result = []

for num in range(1000, 2001):
    if num % 7 == 0 and num % 5 != 0: 
        result.append(num)

print("Numbers divisible by 7 and not divisible by 5:", result)