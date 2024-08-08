str = input("Enter a string: ")


dic = dict()

for char in str:
    if char in dic:
        dic[char] = dic[char] + 1
    else:
        dic[char] = 1
    
print(dic)