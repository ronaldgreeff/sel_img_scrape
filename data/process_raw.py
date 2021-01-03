from PIL import Image
import os

for gender in ['boy', 'girl']:

    proc_gen_dir = os.path.join('processed', gender)
    if not os.path.exists(proc_gen_dir):
        os.makedirs(proc_gen_dir)

    gen_dir = os.path.join('raw', gender)

    for method in os.listdir(gen_dir):

        proc_method_dir = os.path.join(proc_gen_dir, method)
        if not os.path.exists(proc_method_dir):
            os.makedirs(proc_method_dir)

        method_dir = os.path.join(gen_dir, method)

        for imgf in os.listdir(method_dir):
            filename, ext = os.path.splitext(imgf)

            # make everything a jpg. By rewriting all of the images,
            # those that were corrupted with bad end chars can be fixed

            try:
                img = Image.open(os.path.join(os.getcwd(), method_dir, imgf))
            except Exception as e:
                print("Can't open {} {} {} - {}\n".format(gender, method, imgf, e))

            try:
                rgb = img.convert('RGB')
            except Exception as e:
                print("Can't convert {} {} {} - {}\n".format(gender, method, imgf, e))

            rgb.save(os.path.join(os.getcwd(), proc_method_dir, '{}.jpg'.format(filename)), 'JPEG')
