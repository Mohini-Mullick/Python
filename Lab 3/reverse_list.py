def reverse_list(list):
    reversed_lst = []
    for item in list:
        reversed_lst.insert(0, item)
    return reversed_lst

my_list = [1, 'apple', 3.14, 'banana']

reversed_list = reverse_list(my_list)

print(reversed_list)