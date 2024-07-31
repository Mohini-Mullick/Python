def generate_primes(n):
    primes = []
    if n >= 2:
        primes.append(2)
    for num in range(3, n + 1, 2):
        if all(num % i != 0 for i in range(2, int(num**0.5) + 1)):
            primes.append(num)
    return primes

n = int(input("Enter a number: "))
print(f"Prime numbers up to {n}:", generate_primes(n))
