from PIL import Image

# Manually specify the paths to your images
image_files = [
    "/home/aysu//Pictures/Screenshots/1.png",
    "/home/aysu//Pictures/Screenshots/2.png",
    "/home/aysu//Pictures/Screenshots/3.png",
]

# Open the images
images = []
for file in image_files:
    try:
        img = Image.open(file)
        images.append(img)
    except IOError:
        print(f"Error opening file {file}")

# Check if there are images to create a GIF
if len(images) > 0:
    # Save as a GIF
    images[0].save("output4.gif", save_all=True, append_images=images[1:], duration=500, loop=0)
    print("GIF created successfully.")
else:
    print("No images found to create GIF.")
