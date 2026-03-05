import numpy as np
import sys

def read_pgm(filename):
    """Reads a P5 PGM image file manually robustly."""
    with open(filename, 'rb') as f:
        buffer = f.read()
    
    # We need to parse 4 tokens: P5, width, height, maxval
    tokens = []
    idx = 0
    length = len(buffer)
    
    while len(tokens) < 4 and idx < length:
        # 1. Skip whitespace
        while idx < length and (chr(buffer[idx]).isspace()):
            idx += 1
        
        # 2. Check for comments (start with #)
        if idx < length and chr(buffer[idx]) == '#':
            while idx < length and chr(buffer[idx]) != '\n':
                idx += 1
            continue # Go back to skipping whitespace
        
        # 3. Read token
        start = idx
        while idx < length and not (chr(buffer[idx]).isspace()) and chr(buffer[idx]) != '#':
            idx += 1
        
        token = buffer[start:idx]
        if token:
            tokens.append(token)
            
    if len(tokens) < 4:
        raise ValueError("File header incomplete or invalid PGM.")

    if tokens[0] != b'P5':
        raise ValueError("Not a P5 PGM file.")

    width = int(tokens[1])
    height = int(tokens[2])
    max_val = int(tokens[3])

    # The image data starts exactly one char after the max_val token (usually a newline)
    if idx < length and chr(buffer[idx]).isspace():
        idx += 1

    # Read the pixel data
    data = np.frombuffer(buffer[idx:], dtype=np.uint8)
    
    # Safety check: if data size doesn't match, try to align from the end
    expected_size = width * height
    if data.size != expected_size:
        print(f"Warning: Expected {expected_size} pixels, got {data.size}. Attempting auto-fix.")
        if data.size > expected_size:
            data = data[:expected_size]
        else:
            # If we missed the start, try reading last N bytes
            data = np.frombuffer(buffer[-expected_size:], dtype=np.uint8)

    return data.reshape((height, width))

def write_pgm(filename, image):
    """Writes a numpy array as a P5 PGM file."""
    height, width = image.shape
    with open(filename, 'wb') as f:
        header = f"P5\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))
        f.write(image.astype(np.uint8).tobytes())

def pad_image(image, pad_width):
    """Pads the image to handle borders (reflecting edge pixels)."""
    return np.pad(image, pad_width, mode='edge')

def manual_erode(image, kernel_size=3):
    """
    Manual implementation of Erosion.
    For every pixel, replace it with the MINIMUM value in its neighborhood.
    """
    height, width = image.shape
    pad = kernel_size // 2
    padded_image = pad_image(image, pad)
    output = np.zeros_like(image)

    # Sliding window
    for y in range(height):
        for x in range(width):
            # Extract the 3x3 neighborhood
            neighborhood = padded_image[y:y+kernel_size, x:x+kernel_size]
            # Erosion = Min
            output[y, x] = np.min(neighborhood)
    return output

def manual_dilate(image, kernel_size=3):
    """
    Manual implementation of Dilation.
    For every pixel, replace it with the MAXIMUM value in its neighborhood.
    """
    height, width = image.shape
    pad = kernel_size // 2
    padded_image = pad_image(image, pad)
    output = np.zeros_like(image)

    # Sliding window
    for y in range(height):
        for x in range(width):
            # Extract the 3x3 neighborhood
            neighborhood = padded_image[y:y+kernel_size, x:x+kernel_size]
            # Dilation = Max
            output[y, x] = np.max(neighborhood)
    return output

def solve_exercise_manual():
    print("Reading image...")
    try:
        I = read_pgm('isn_256.pgm')
    except FileNotFoundError:
        print("Error: isn_256.pgm not found.")
        return

    print("Computing Filter 1: Opening (Erosion -> Dilation)...")
    eroded = manual_erode(I)
    opening_I = manual_dilate(eroded)  # Filter 1
    
    print("Computing Filter 2: Closing (Dilation -> Erosion)...")
    dilated = manual_dilate(I)
    closing_I = manual_erode(dilated)  # Filter 2

    print("Computing Filter 3: Closing of Opening...")
    # Apply Closing (Dilation -> Erosion) on Filter 1
    f3_dilated = manual_dilate(opening_I)
    filter_3 = manual_erode(f3_dilated)

    print("Computing Filter 4: Opening of Closing...")
    # Apply Opening (Erosion -> Dilation) on Filter 2
    f4_eroded = manual_erode(closing_I)
    filter_4 = manual_dilate(f4_eroded)

    # Save outputs
    write_pgm('exercise_08a_output_filter1.pgm', opening_I)
    write_pgm('exercise_08a_output_filter2.pgm', closing_I)
    write_pgm('exercise_08a_output_filter3.pgm', filter_3)
    write_pgm('exercise_08a_output_filter4.pgm', filter_4)

    # Generate Text Output
    output_text = (
        "3\n"
        "4\n"
        "Explanation: Salt-and-pepper noise adds both white pixels (salt) and black pixels (pepper). "
        "Opening removes white pixels (salt). Closing removes black pixels (pepper). "
        "To clean the image completely, we must use a sequence that addresses both. "
        "Filter 3 (Closing of Opening) first removes salt, then removes pepper. "
        "Filter 4 (Opening of Closing) first removes pepper, then removes salt. "
        "Filters 1 and 2 only remove half of the noise types."
    )
    
    with open('exercise_08a_output_01.txt', 'w') as f:
        f.write(output_text)

    print("Done. Results saved.")

if __name__ == "__main__":
    solve_exercise_manual()
