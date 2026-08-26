import cv2
import os
import numpy as np
import random
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageEnhance


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def gaussian(img):
    kernel_5x5 = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ])
    kernel_5x5 = kernel_5x5 / kernel_5x5.sum()
    k5 = ndimage.convolve(img, kernel_5x5)
    return k5


def points_extend(points, degree=0.02):
    top_point = points[0]
    left_base = points[1]
    right_base = points[2]

    vec_top = np.array([0, -1])
    vec_left = np.array([-1, 0])
    vec_right = np.array([1, 0])

    original_height = abs(top_point[1] - left_base[1])
    expand_pixels = int(original_height * degree)

    new_top = (top_point[0] + vec_top[0] * expand_pixels, top_point[1] + vec_top[1] * expand_pixels)
    new_left = (left_base[0] + vec_left[0] * expand_pixels, left_base[1] + vec_left[1] * expand_pixels // 2)
    new_right = (right_base[0] + vec_right[0] * expand_pixels, right_base[1] + vec_right[1] * expand_pixels // 2)

    new_points = np.array([new_top, new_left, new_right], np.int32)
    return new_points


# Simulate eyelash-like artifacts
def eyelash_simulation(img, mask, image_size):
    if img.shape[0] != image_size[0] or img.shape[1] != image_size[1]:
        img = cv2.resize(img, image_size)

    h, w, c = img.shape

    mask_A = mask[0]
    mask_A_3 = mask_A / mask_A.max()
    mask_A_3 = mask_A_3[:, :, np.newaxis]

    wp = random.randint(int(-w * 0.2), int(w * 0.2))
    hp = random.randint(int(-h * 0.3), int(h * 0.3))
    center = (w // 2 + wp, h // 2 + hp)

    transmap = np.ones(shape=[h, w], dtype=np.float32)
    if 0 <= center[1] < h and 0 <= center[0] < w:
        transmap[center[1], center[0]] = 0

    distance = ndimage.distance_transform_edt(transmap)
    transmap = gaussian_filter(distance, sigma=5) * mask_A
    transmap = transmap / transmap.max()

    # eyelash-like shadow
    eyelash_mask = np.zeros((h, w), dtype=np.float32)
    eyelash_mask_1 = np.zeros_like(eyelash_mask)

    num_eyelashes = random.randint(22, 32)
    bin_width = (w - 20) / num_eyelashes

    depth = [random.uniform(-0.15, -0.05)]

    for i in range(num_eyelashes):
        bin_start = 10 + i * bin_width
        bin_end = bin_start + bin_width
        if bin_end > w - 10:
            bin_end = w - 10
        x = random.randint(int(bin_start), int(bin_end))
        dev = random.uniform(w * 0.04, w * 0.1)

        if x < w * 0.3:
            height = random.randint(int(h * 0.25), int(h * 0.45))
            if x < w * 0.2:
                height = random.randint(int(h * 0.45), int(h * 0.6))
            dire = 1
        elif x > w * 0.7:
            height = random.randint(int(h * 0.25), int(h * 0.45))
            if x > w * 0.8:
                height = random.randint(int(h * 0.45), int(h * 0.6))
            dire = -1
        else:
            height = random.randint(int(h * 0.15), int(h * 0.55))
            dire = random.choice([-1, 1])

        width = random.randint(5, 14)

        y_foot = h - 1 if center[1] <= h // 2 else 1

        top_point = [x + dev * dire, y_foot - height] if center[1] <= h // 2 else [x + dev * dire, y_foot + height]
        left_base = [x - width, y_foot]
        right_base = [x + width, y_foot]

        points = np.array([top_point, left_base, right_base], np.int32)
        points_2 = points_extend(points,0.02)

        if depth[i] >= -0.7:
            depth.append(depth[i]+random.uniform(-0.06, -0.02))
        else:
            depth.append(depth[i])

        cv2.fillPoly(eyelash_mask_1, [points_2], -depth[i])
        cv2.fillPoly(eyelash_mask, [points], depth[i])

        eyelash_edge = eyelash_mask_1 + eyelash_mask

    eyelash_mask = gaussian_filter(eyelash_mask, sigma=5)
    eyelash_edge = gaussian_filter(eyelash_edge, sigma=3)

    transmap += eyelash_mask * 0.5
    transmap += eyelash_edge * 0.8
    # transmap = np.clip(transmap, 0, 1)

    sum_map = transmap

    B, G, R = cv2.split(img)
    panel = cv2.merge([sum_map * (B.max() - B), sum_map * (G.max() - G), sum_map * (R.max() - R)])

    panel_ratio = random.uniform(0.4, 0.8)
    sum_degrad = 0.8 * img + panel * panel_ratio
    sum_degrad[sum_degrad > 255] = 255

    c = random.uniform(0.9, 1.3)
    b = random.uniform(0.9, 1.0)
    e = random.uniform(0.9, 1.3)
    img = Image.fromarray(sum_degrad.astype('uint8'))

    enh_con = ImageEnhance.Contrast(img).enhance(c)
    enh_bri = ImageEnhance.Brightness(enh_con).enhance(b)
    enh_col = ImageEnhance.Color(enh_bri).enhance(e)

    img_de = np.array(enh_col).astype('float32') * mask_A_3

    return img_de

