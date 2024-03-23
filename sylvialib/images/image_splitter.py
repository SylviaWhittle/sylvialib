"""Splits an image into multiple sprites"""

import sys
from pathlib import Path
from PIL import Image

# Ask for the image path and check that it exists
image_path = Path(
    input("Enter the path to the image (including file name and extension): ")
)
if not image_path.exists():
    print(f"Error: {image_path} does not exist, exiting.")
    # print current directory
    print(f"Current directory: {Path.cwd()}")
    sys.exit()

image_name = image_path.stem
image = Image.open(image_path)
width, height = image.size

print(f"Image {image_name} dimensions: {width}x{height}")

# Ask where to output the sprites and create the directory if it doesn't exist
output_path = Path(
    input(
        "(Optional) Enter the path to the output directory: (default: ./split_images/)"
    )
)
if output_path == Path("") or output_path == Path("."):
    output_path = Path("./split_images/")
print(f"Output directory: {output_path}")
if not output_path.exists():
    print(f"Creating {output_path}")
    output_path.mkdir(exist_ok=True)

# Ask for the sprite size and padding
sprite_width = int(input("Enter the sprite width: "))
sprite_height = int(input("Enter the sprite height: "))
sprite_padding = int(input("Enter the sprite padding: "))

# If the image is already the correct size, just save it
if width == sprite_width and height == sprite_height:
    print("Image is already the correct size. Save it? (y/n)")
    if input().lower() == "y":
        image.save(output_path / "0.png")
        print(f"Saved {image_path} as 0.png")
        print("Enter any key to exit.")
        input()
    else:
        sys.exit()
else:
    # Check that the image is a size that is a multiple of the sprite size plus padding
    if (
        width % (sprite_width + sprite_padding) != 0
        or height % (sprite_height + sprite_padding) != 0
    ):
        print(
            f"""Image dimensions are not a multiple of the sprite size plus padding.
\nImage width: {width}, image height: {height}, sprite width: {sprite_width}, 
sprite height: {sprite_height}, sprite padding: {sprite_padding}, closest multiple of
sprite size: {width // sprite_width * sprite_width}x{height // sprite_height * sprite_height}
remainder: {width % sprite_width}x{height % sprite_height}"""
        )
        print("exiting...")
        input()
        sys.exit()

    # If the number of sprites is greater than 32, ask for confirmation
    sprite_count = (width // (sprite_width + sprite_padding)) * (
        height // (sprite_height + sprite_padding)
    )
    if sprite_count > 32:
        print(
            f"Up to {sprite_count} sprites will be created (blank ones will be ignored). Continue? (y/n)"
        )
        if input().lower() != "y":
            sys.exit()

    # Split the image into sprites
    sprites = []
    for y in range(0, height, sprite_height + sprite_padding):
        for x in range(0, width, sprite_width + sprite_padding):

            sprite = image.crop((x, y, x + sprite_width, y + sprite_height))

            # Check if the sprite is empty (all pixels are transparent, ie have an alpha value of 0)
            if sprite.getbbox() is None:
                continue

            sprites.append(sprite)

    print(f"Found {len(sprites)} non-blank sprites. Save them?")
    if input().lower() != "y":
        sys.exit()

    # Save the sprites
    for sprite_index, sprite in enumerate(sprites):
        sprite.save(output_path / f"{image_name}_{sprite_index}.png")

    print(f"Split {image_path} into {len(sprites)} sprites")
    print("Enter any key to exit.")
    input()
