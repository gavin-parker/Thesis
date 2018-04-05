import bpy
import os
import glob
from os.path import isdir, join
import sys
import random
import mathutils
from fuzzywuzzy import fuzz, process
from contextlib import contextmanager
import addon_utils
import math
scene_dir = os.environ['SCENE_DIR']

def main():
    print("hi")
    bpy.ops.mesh.primitive_cube_add()
    normal_mat = bpy.data.materials['work_please']
    objects = bpy.context.selected_objects[:]
    for obj in objects:
        if obj.type != 'CAMERA' and obj.type != 'LAMP':
            if obj.data.materials:
                obj.data.materials[0] = normal_mat
            else:
                obj.data.materials.append(normal_mat)

            obj.active_material = normal_mat
            bpy.context.scene.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.object.material_slot_assign()
            bpy.ops.object.mode_set(mode='OBJECT')
            obj.select = True
    bpy.data.scenes['Scene'].render.filepath = "test.png"
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    main()
