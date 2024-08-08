str = input("Enter a comma separated words: ")

words = str.split(",")

sorted_words = sorted(words, key=lambda x: x.lower())


print("Sorted words:", ",".join(sorted_words))