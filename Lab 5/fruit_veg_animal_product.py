# 1. Create fruits, vegetables, and animal products tuples
fruits = ("apple", "banana", "orange")
vegetables = ("carrot", "broccoli", "spinach")
animal_products = ("milk", "egg", "cheese")
food_stuff_tp = fruits + vegetables + animal_products
print("food_stuff_tp:", food_stuff_tp)
food_stuff_lt = list(food_stuff_tp)
print("food_stuff_lt:", food_stuff_lt)
mid_index = len(food_stuff_lt) // 2
if len(food_stuff_lt) % 2 == 0:
    middle_items = food_stuff_lt[mid_index - 1: mid_index + 1]
else:
    middle_item = food_stuff_lt[mid_index]
print("Middle item(s):", middle_items if len(food_stuff_lt) % 2 == 0 else middle_item)
first_three = food_stuff_lt[:3]
last_three = food_stuff_lt[-3:]
print("First three items:", first_three)
print("Last three items:", last_three)
del food_stuff_tp
