from degrad_de import *
from cataract_simulation import *
import random
import shutil
import cv2


random.seed(2025)
np.random.seed(2025)


def mkdirs(*dir_names):
    for dir_name in dir_names:
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)


def generate_type_list(num_type):
    type_list = []
    if num_type >= len(type_map):
        for i in range(num_type):
            t = random.randint(0, len(type_map) - 1)
            type_list.append(type_map[t])
    else:
        for i in range(num_type):
            t = random.randint(0, len(type_map) - 1)
            type_list.append(type_map[t])
    return type_list


def degradation(image_dir, image_mask_dir, output_dir, num_type=16, concatenate=True):
    image_list = os.listdir(image_dir)
    for image_name in image_list:
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(image_mask_dir, image_name)

        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)

        # shutil.copy(mask_path, os.path.join(output_mask_dir, image_name))
        shutil.copy(mask_path, os.path.join(output_mask_dir, image_name))
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((sizeX, sizeY), Image.BICUBIC)
        mask = np.expand_dims(mask, axis=2)
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0

        type_list = generate_type_list(num_type)
        for i, t in enumerate(type_list):
            dst_img = os.path.join(output_dir, '{}-{}.png'.format(image_name.split('.')[0], i))

            shutil.copy(mask_path, os.path.join(output_mask_dir, '{}-{}.png'.format(image_name.split('.')[0], i)))

            # single cataract simulation
            if t == '00001':
                cataract_image, clear_image = cataract_simulation(image_path, mask, (sizeX, sizeY))
                if concatenate:
                    im_AB = np.concatenate([cataract_image, clear_image], 1)
                    cv2.imwrite(dst_img, im_AB)
                else:
                    cv2.imwrite(dst_img, cataract_image)
                continue

            #  other four types of simulations
            else:
                t = t[:4]
                r_img, r_params = DE_process(img, mask, sizeX, sizeY, t)
                r_img = np.clip(r_img, 0, 255).astype(np.uint8)
                if concatenate:
                    im_AB = np.concatenate([r_img, img_np], 1)
                    imwrite(dst_img, im_AB)
                else:
                    imwrite(dst_img, r_img)


if __name__ == "__main__":
    sizeX = 512
    sizeY = 512

    type_map = ['00001',
                '00010',
                '00100', '00110',
                '01000', '01010', '01100', '01110',
                '10000', '10100', '11000', '11100',
                '00011', '00101', '01001', '01101', '01111',
                ]
    """
        '00001': CATARACT
        '00010': DE_ILLUMINATION
        '00100': DE_SPOT
        '01000': DE_BLUR
        '10000': EYELASH
    """
    num_type = 16
    image_root = '../../images/preprocess_dataset'
    clear_image_dir = os.path.join(image_root, 'image')
    clear_image_mask_dir = os.path.join(image_root, 'mask')

    output_dir = os.path.join(image_root, 'low_quality_image')
    output_mask_dir = os.path.join(image_root, 'low_quality_mask')
    mkdirs(output_dir, output_mask_dir)

    degradation(clear_image_dir, clear_image_mask_dir, output_dir, num_type=num_type, concatenate=True)

    print('Finished!')
