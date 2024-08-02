def print_power_table(rows):
    for i in range(1, rows + 1):
        line = [str(i)]
        for j in range(5):
            line.append(str(i ** j))
        print(" ".join(line))

# Test case with 5 rows as specified
print_power_table(5)
