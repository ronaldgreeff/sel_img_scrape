import os
import csv
from PIL import Image

for folder in os.listdir():
    fp = os.path.join(os.getcwd(), folder)

    if os.path.isdir(fp):
        bn = os.path.basename(fp)

        if bn != '0' and bn != 'full':

            for gender in os.listdir(fp):
                    for gen_fold in os.listdir(os.path.join(fp, gender)):

                        if gen_fold != 'multi':

                            for image_name in os.listdir(os.path.join(fp, gender, gen_fold)):

                                filename, ext = os.path.splitext(image_name)

                                try:
                                    img = Image.open(os.path.join(fp, gender, gen_fold, image_name))
                                except Exception as e:
                                    print("Can't open {} - {}\n".format(image_name, e))

                                try:
                                    rgb = img.convert('RGB')
                                except Exception as e:
                                    print("Can't convert {} - {}\n".format(image_name, e))

                                rgb.save(os.path.join(os.getcwd(), 'full', gen_fold, '{}_{}.jpg'.format(bn, filename)), 'JPEG')
