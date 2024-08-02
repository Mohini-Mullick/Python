def print_spiral_pattern(N):
    # Initialize an N x N grid with zeros
    grid = [[0] * N for _ in range(N)]
    
    # Define initial positions and directions
    top, bottom = 0, N - 1
    left, right = 0, N - 1
    num = 1
    
    while num <= N * N:
        # Traverse from left to right along the top row
        for i in range(left, right + 1):
            grid[top][i] = num
            num += 1
        top += 1
        
        # Traverse from top to bottom along the right column
        for i in range(top, bottom + 1):
            grid[i][right] = num
            num += 1
        right -= 1
        
        # Traverse from right to left along the bottom row
        for i in range(right, left - 1, -1):
            grid[bottom][i] = num
            num += 1
        bottom -= 1
        
        # Traverse from bottom to top along the left column
        for i in range(bottom, top - 1, -1):
            grid[i][left] = num
            num += 1
        left += 1
    
    # Print the grid in the specified pattern
    for row in grid:
        print(" ".join(map(str, row)))

# Test cases
print("N=2")
print_spiral_pattern(2)
print()

print("N=3")
print_spiral_pattern(3)
print()

print("N=4")
print_spiral_pattern(4)
