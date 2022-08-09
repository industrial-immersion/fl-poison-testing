#!/usr/bin/env python3

from __future__ import annotations

import glob
import os
import random
from itertools import permutations

from PIL import Image  # type: ignore


def poison_data(
    num_to_poison: int = 10,
    path: str = "/root/data/cinic-10/",
    amt_of_pixels: int = 100,
    acceptable_range: float = 0.05,
) -> None:
    fldr_list = os.listdir(path=path + "train/")

    pixel_list = list(set(permutations(range(32), 2)))
    for fldrs in fldr_list:
        to_poison = random.sample(
            glob.glob(path + "train/" + fldrs + "/*.png"), num_to_poison
        )
        # Create the directories if needed.
        if not os.path.isdir(path + "train_poison/" + fldrs):
            os.makedirs(path + "/train_poison/" + fldrs)
        for pics in to_poison:
            img = Image.open(pics)
            pixel_map = img.load()
            pixels_to_change = random.sample(pixel_list, k=amt_of_pixels)
            for pixel in pixels_to_change:
                # Select the pixel to change.
                x = pixel[0]
                y = pixel[1]
                # Change each selected pixel by altering each channel by the range.
                if not isinstance(pixel_map[x, y], int):
                    pixel_map[x, y] = (
                        int(
                            random.uniform(
                                pixel_map[x, y][0] * (1 - acceptable_range),
                                pixel_map[x, y][0] * (1 + acceptable_range),
                            )
                        ),
                        int(
                            random.uniform(
                                pixel_map[x, y][1] * (1 - acceptable_range),
                                pixel_map[x, y][1] * (1 + acceptable_range),
                            )
                        ),
                        int(
                            random.uniform(
                                pixel_map[x, y][2] * (1 - acceptable_range),
                                pixel_map[x, y][2] * (1 + acceptable_range),
                            )
                        ),
                    )

                # If the image is greyscale we instead change the intensity.
                else:
                    pixel_map[x, y] = int(
                        random.uniform(
                            pixel_map[x, y] * (1 - acceptable_range),
                            pixel_map[x, y] * (1 + acceptable_range),
                        )
                    )
            # Generate the new filename.
            new_name = img.filename.split("/")[-1].split(".")[0] + "_poisoned.png"
            # Save the new image to the selected folder.
            img.save(path + "train_poison/" + fldrs + "/" + new_name)


if __name__ == "__main__":
    poison_data()
