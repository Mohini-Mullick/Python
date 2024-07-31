first_term = 2
common_ratio = 3
num_terms = 6

current_term = first_term

for _ in range(num_terms):
    print(current_term, end=" ")
    current_term *= common_ratio