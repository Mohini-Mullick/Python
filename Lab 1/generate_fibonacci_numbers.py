def fibonacci_up_to_n(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    fib_series = [0, 1]
    while True:
        next_fib = fib_series[-1] + fib_series[-2]
        if next_fib > n:
            break
        fib_series.append(next_fib)
    
    return fib_series

# Example usage:
n = int(input("Enter a number: "))
print(fibonacci_up_to_n(n))
