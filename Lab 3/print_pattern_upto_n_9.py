def print_pattern(N):
    print(".")

    for i in range(N - 2):
        print(f"/{' '*i}\\")
    
    print("/" + "_" * (N) + "\\")

print("N=2")
print_pattern(2)

print("N=3")
print_pattern(3)

print("N=4")
print_pattern(4)