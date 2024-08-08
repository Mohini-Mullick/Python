str = input("Enter a space separated words: ")

words = str.split()

deduplicated_words = list(set(words))

sorted_words = sorted(deduplicated_words, key=lambda x: x)

print("Sorted unique words:", " ".join(sorted_words))