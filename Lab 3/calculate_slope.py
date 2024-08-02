def calculate_slope(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    return slope

x1, y1 = 1, 2
x2, y2 = 4, 6

slope = calculate_slope(x1, y1, x2, y2)

print(f'The slope of the line passing through ({x1}, {y1}) and ({x2}, {y2}) is {slope}')