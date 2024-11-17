""" animation utilities """

import os

import imageio


def merge_images_into_gif(image_folder, gif_name, duration=0.01):
    """creates a GIF file from a series of animation frames"""
    with imageio.get_writer(gif_name, mode="I", duration=duration, loop=0) as writer:
        for filename in sorted(os.listdir(image_folder)):
            image = imageio.v3.imread(os.path.join(image_folder, filename))
            writer.append_data(image)
