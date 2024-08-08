str = input("Enter a string: ")

letters, digits = 0, 0

for char in str:
    if char.isalpha():
        letters += 1
    elif char.isdigit():
        digits += 1
        
print(f"Number of letters: {letters}")

print(f"Number of digits: {digits}")