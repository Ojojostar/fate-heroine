import cv2
import os
import sys
from glob import glob


def renamer(src, dst, name):

    files = [y for x in os.walk(src) for y in glob(os.path.join(x[0], '*.*'))]
    counter = 0

    for image_file in files:
        counter += 1
        target_path = "/".join(image_file.strip("/").split('/')[1:-1])
        target_path = os.path.join(dst, target_path) + "/"
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        image = cv2.imread(image_file)
        filename = f'{name}_{counter}'
        cv2.imwrite(
            os.path.join(target_path, filename + ".png"), image
        )


def main():
    name1 = 'rin'
    renamer(f'train/{name1}', f'testingtestingtesting/{name1}', name1)

if __name__ == '__main__':
    main()
# python bulk_convert.py raw/Rin cropped/Rin
