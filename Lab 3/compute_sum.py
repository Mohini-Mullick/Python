def compute_series_sum(n):
    sum = 0
    for i in range(1, n+1):
        if i % 2 == 0:
            sum -= 1 / i
        else:
            sum += 1 / i
    return sum

n = 10
series_sum = compute_series_sum(n)
print(f"The sum of the series up to {n} terms is {series_sum}")