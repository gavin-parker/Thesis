import os
from multiprocessing import Pool, cpu_count
import sys


def render(prefix):
    os.system('/home/gavin/blender-2.79-linux-glibc219-x86_64/blender canvas2.blend --background -E CYCLES --python {}/random_object.py prefix={}'.format(os.getcwd(), prefix))


if __name__ == "__main__":
    cores = cpu_count()
    id = 0
    print("using {} cores".format(cores))
    for arg in sys.argv:
        if 'arr' in arg:
            id = int(arg.split('=')[1])

    for i in range(100):
        render(id*1000 + i)