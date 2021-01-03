# some of the jpgs get corrupted during transfer, others
# are just uploaded truncated. This detects those corruptions

import os

to_remove = []

for gender in ['boy', 'girl']:

    gen_dir = os.path.join('processed', gender)

    for method in os.listdir(gen_dir):
        method_dir = os.path.join(gen_dir, method)

        for imgf in os.listdir(method_dir):

            with open(os.path.join(method_dir, imgf), 'rb') as f:
                check_chars = f.read()[-2:]

            # check if jpg is corrupt
            if check_chars != b'\xff\xd9':
                print('{} {}'.format(gender, file))
                # to_remove.append(os.path.join(gen_dir, file))

# for r in to_remove:
#     os.remove(r)
