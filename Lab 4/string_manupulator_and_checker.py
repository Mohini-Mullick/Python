import re

str = input("Enter a string to manupulate or check: ")

print("1. Reverse It")
print("2. Palindrome Check")
print("3. Check if it end with a specific substring")
print("4. Capitalize the first letter of each word")
print("5. Check if the string is anagram of another string")
print("6. Remove vowels")
print("7. Length of the longest word in the sentence")
print("8. Exit")


def menu(str):
    choice = int(input("Enter a choice: "))

    match choice:
        case 1:
            print(f"Reversed string is: {str[::-1]}")
        case 2:
            if str == str[::-1]:
                print("The string is a Palindrome")
            else:
                print("The string is not a Palindrome")
        case 3:
            substring = input("Enter substring: ")
    
            if str.endswith(substring):
                print(f"The string ends with the substring \"{substring}\"")
            else:
                print(f"The string does not end with the substring \"{substring}\"")
        case 4:
            print(f"Every word in capital: {str.title()}")
        case 5:
            another_str = input("Enter another string: ")
            if len(str) != len(another_str):
                print("The strings are not Anagrams")
            else:
                ar = [0] * 26
                str = str.lower()
                another_str = another_str.lower()
                for i in str:
                    ar[ord(i) - 97] += 1
                for i in another_str:
                    ar[ord(i) - 97] -= 1
                flag = 0
                for i in range(26):
                    if ar[i] != 0:
                        print("The strings are not Anagrams")
                        flag = 1
                        break
                if not flag:
                    print("The strings are Anagrams")
        case 6:
            result = ""
            vowels = "aeiouAEIOU"
            for i in str:
                if i not in vowels:
                    result += i
            print(f"String without vowels: {result}")
        case 7:
            words = str.split(" ")
            result = ""
            longest_length = 0
            for i in words:
                if len(i) > longest_length:
                    longest_length = len(i)
                    result = i
            print(f"Longest word in the string: {result}")
        case 8:
            exit()
    menu(str)
            
menu(str)