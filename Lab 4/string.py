str = input("Give a string: ")

print("String length:", len(str))

words = str.strip().lower().split(" ")

country = words.index("country")

if country!= -1:
    print("The word 'country' is found at index:", country)
else:
    print("'country' is not found in the given string.")

dic = dict()

for word in words:
    if word in dic:
        dic[word] = dic[word] + 1
    else:
        dic[word] = 1
        
print("Word count:", dic)

