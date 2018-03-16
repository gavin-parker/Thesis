import os
os.system('blender canvas2.blend --background -E CYCLES --python {}/random_object.py'.format(os.getcwd()))
