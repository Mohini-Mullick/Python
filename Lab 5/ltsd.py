
ages = [10, 22, 19, 24, 20, 25, 26, 24, 25, 24]
ages.sort()
min_age = ages[0]
max_age = ages[-1]
print("Sorted ages:", ages)
print("Min age:", min_age)
print("Max age:", max_age)
ages.append(min_age)
ages.append(max_age)
print("Ages after adding min and max again:", ages)
n = len(ages)
if n % 2 == 0:
    median_age = (ages[n//2 - 1] + ages[n//2]) / 2
else:
    median_age = ages[n//2]
print("Median age:", median_age)
average_age = sum(ages) / len(ages)
print("Average age:", average_age)
age_range = max_age - min_age
print("Age range:", age_range)
min_diff = abs(min_age - average_age)
max_diff = abs(max_age - average_age)
print("Absolute difference between min and average age:", min_diff)
print("Absolute difference between max and average age:", max_diff)
