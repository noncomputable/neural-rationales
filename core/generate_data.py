from PIL import Image, ImageDraw
import os
import pandas as pd
import math
import numpy as np

def get_line_coords(n_lines, angle, max_coord):
    angle = angle * math.pi / 180
    coords = np.random.randint(max_coord, size = (n_lines, 4))
    for coord in coords:
        x1, y1 = coord[0], coord[1]
        x2_dist, y2_dist = np.random.randint(x1, max_coord), np.random.randint(y1, max_coord)
        direction = 1 if np.random.random() < .5 else -1
        coord[2] = coord[0] + direction * math.sin(angle) * x2_dist
        coord[3] = coord[1] + direction * math.cos(angle) * y2_dist
    return coords

def draw_lines(img_draw, angle, max_coord):
    n_lines = np.random.randint(3, 10)
    line_coords = get_line_coords(n_lines, angle, max_coord)
    colors = np.random.randint(256, size = (n_lines, 3))
    for (x1, y1, x2, y2), (r, g, b) in zip(line_coords, colors):
        img_draw.line((x1, y1, x2, y2), fill=(r,g,b), width=2)

def get_line_img(angle, bg_color, img_size):
    img = Image.new("RGB", (img_size, img_size), bg_color)
    img_draw = ImageDraw.Draw(img)
    draw_lines(img_draw, angle, img_size)

    return img

def make_line_imgs(angles, num_per_angle, img_size, bg_color, path):
    """
    Args:
    angles - List of angles to orient the lines.
    num_per_angle - Number of images to generate per angle.
    img_size - The pixel width=height of the images.
    bg_color - 
    path - 
    """

    class_annotations = {"class_name": angles}
    img_annotations = {"img_class": [], "class_img_idx": []}

    n_images = len(angles) * num_per_angle
    
    if bg_color != "random":
        bg_colors = np.full((n_images, 3), bg_color)
    else:
        bg_colors = np.random.randint(256, size = (n_images, 3))
    
    for i, angle in enumerate(angles):
        os.mkdir(f"{path}/{angle}")
        for j in range(num_per_angle):
            img_annotations["img_class"].append(i)
            img_annotations["class_img_idx"].append(j)
            bg_color = tuple(bg_colors[num_per_angle*i + j])
            img = get_line_img(angle, bg_color, img_size)
            img.save(f"{path}/{angle}/{angle}_{j}.png")

    pd.DataFrame(class_annotations).to_csv(f"{path}/class_names.csv", index = False, header = True)
    pd.DataFrame(img_annotations).to_csv(f"{path}/img_annotations.csv", index = False, header = True)
