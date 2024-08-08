str = input("Enter a string: ")

list = []

for char in str:
    if char.isdigit():
        list.append(int(char))
    
print(list)