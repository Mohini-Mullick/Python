def print_segment_display(number, N):
    segments = {
        0: [" _ ", "| |", "|_|", "   ", " _ ", "| |", "|_|", "   "],
        1: ["   ", "  |", "  |", "   ", "   ", "  |", "  |", "   "],
        2: [" _ ", " _|", "|_ ", "   ", " _ ", "|_ ", " _|", "   "],
        3: [" _ ", " _|", " _|", "   ", " _ ", " _|", " _|", "   "],
        4: ["   ", "|_|", "  |", "   ", "   ", "  |", "|_|", "   "],
        5: [" _ ", "|_ ", " _|", "   ", " _ ", " _|", "|_ ", "   "],
        6: [" _ ", "|_ ", "|_|", "   ", " _ ", "| |", "|_|", "   "],
        7: [" _ ", "  |", "  |", "   ", "   ", "  |", "  |", "   "],
        8: [" _ ", "|_|", "|_|", "   ", " _ ", "|_|", "|_|", "   "],
        9: [" _ ", "|_|", " _|", "   ", " _ ", "  |", "|_|", "   "],
    }
    
    result = []
    for line in range(N):
        display_lines = []
        for digit in str(number):
            display_lines.append(segments[int(digit)][line])
        result.append(" ".join(display_lines))
    
    for line in result:
        print(line)

# Test cases
print("N=2")
print_segment_display(2, 3)
print()

print("N=3")
print_segment_display(3, 3)
print()

print("N=4")
print_segment_display(4, 4)
