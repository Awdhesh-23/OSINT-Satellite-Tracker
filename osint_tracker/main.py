import cv2

# 1. Define the relative path to your specific image in the data folder
image_path = "data/test_airport.png"

# 2. Command OpenCV to read the file and convert it to a mathematical matrix
image_matrix = cv2.imread(image_path)

# 3. Defensive programming: Check if the matrix is empty (file not found)
if image_matrix is None:
    print(f"CRITICAL ERROR: OpenCV could not find or read the image at {image_path}")
    print("Check your folder structure and file path!")
else:
    # 4. Prove the math engine sees the data by printing its dimensions
    print(f"Success! Image loaded. The matrix shape (Height, Width, Channels) is: {image_matrix.shape}")
    
    # 5. Command OpenCV to open a GUI window and render the matrix as an image
    cv2.imshow("OSINT Satellite Feed", image_matrix)
    
    # 6. Pause the script indefinitely until a human presses any key
    cv2.waitKey(0)
    
    # 7. Destroy the window to free up RAM once a key is pressed
    cv2.destroyAllWindows()