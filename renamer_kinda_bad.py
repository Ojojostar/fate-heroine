import cv2
import os
import sys
from glob import glob


def renamer(src, name):

    files = [y for x in os.walk(src) for y in glob(os.path.join(x[0], '*.*'))]
    counter = 0

    for image_file in files:
        counter += 1
        target_path = "/".join(image_file.strip("/").split('/')[1:-1])
        target_path = os.path.join(src, target_path) + "/"

        filename = f'{name}_{counter}'
        image_file.rename(filename)
        os.rename(src + filename, src + name + str(counter) + ".png")

        cv2.imwrite(
            os.path.join(target_path, filename + ".png"), image
        )


def main():
    name1 = 'rin'
    renamer(f'testingtestingtesting/{name1}', name1)

if __name__ == '__main__':
    main()
# python bulk_convert.py raw/Rin cropped/Rin
