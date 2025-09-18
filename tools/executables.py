"""
Scripts for gif conversions.
"""

from PIL import Image, ImageSequence
import os

def main():
    gif_path = "C:\EK Google Drive\School\Comprehensives\Comp I Presentation\img\quadratic_anim_20250917_201205.gif"
    output_path = "C:\EK Google Drive\School\Comprehensives\Comp I Presentation\img\quadratic_frames"
    # gif2png(gif_path, output_path)

    output_path = "C:\EK Google Drive\School\Comprehensives\Comp I Presentation\img\quadratic_anim_20250917_201205.pdf"
    pngs2pdf(gif_path, output_path)


def pngs2pdf(gif_path, output_path):
    frames = []
    with Image.open(gif_path) as im:
        for frame in ImageSequence.Iterator(im):
            frames.append(frame.convert("RGB"))

    # Save all frames as a multi-page PDF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:]
    )

    print(f"Saved {len(frames)} pages to {output_path}")

def gif2png(gif_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    # Open the GIF
    with Image.open(gif_path) as im:
        for i, frame in enumerate(ImageSequence.Iterator(im)):
            # Convert frame to RGBA so it's saved cleanly
            frame = frame.convert("RGBA")
            frame.save(f"{output_path}/frame-{i:03d}.png")

    print(f"Saved {i+1} frames to {output_path}/")

if __name__ == "__main__":
    main()