import bpy
import os
import glob
from os.path import isdir, join
import sys
import random


class SceneGenerator:
    temp_obj_file = '{}/temp/temp.obj'.format(os.getcwd())
    temp_mtl_file = '{}/temp/temp.mtl'.format(os.getcwd())
    scene = bpy.context.scene
    def __init__(self):
        self.scenes, self.textures = find_scene_data()


    def random_render(self):
        scene_type = random.choice(list(self.scenes.values()))
        scene = random.choice(scene_type)
        self.load_scene(scene)
        self.light_scene()
        bpy.data.scenes['Scene'].render.filepath = "renders/{}".format('test.png')
        #bpy.ops.render.render(write_still=True)

    def light_scene(self):
        bpy.data.images.load('{}/envmaps/{}'.format(os.getcwd(), 'gt.hdr'), check_existing=False)
        self.scene.world.use_nodes = True
        self.scene.world.node_tree.nodes['Environment Texture'].image = bpy.data.images['gt.hdr']
        for obj in self.scene.objects:
            if 'light' in obj.name:
                new_light = bpy.context.scene.objects['Lamp'].copy()
                new_light.location = obj.location
                new_light.location.z = 2.5
                self.scene.objects.link(new_light)

    def apply_random_textures(self, mtl):
        new_lines = []
        with open(mtl) as original:
            lines = original.readlines()
            new_lines = lines
            for i, line in enumerate(lines):
                if 'map_Kd' in line:
                    item = line.strip().split('/')[-1]
                    try:
                        texture = random.choice(self.textures[item])
                    except IndexError:
                        print("item {} has no texture!".format(item), file=sys.stderr)
                    new_lines[i] = 'map_Kd {}\n'.format(texture)
        assert len(new_lines) > 0
        return new_lines

    def load_scene(self, scene):
        print("loading {}".format(scene))
        with open(scene[0]) as original:
            lines = original.readlines()
            lines[2] = 'mtllib temp.mtl\n'
        with open(self.temp_obj_file, 'w') as temp_file:
            temp_file.writelines(lines)
        with open(self.temp_mtl_file, 'w') as temp_file:
            temp_file.writelines(self.apply_random_textures(scene[1]))
        bpy.ops.import_scene.obj(filepath=self.temp_obj_file)

def find_scene_data(scene_dir='/home/gavin/SceneNetRGBD_Layouts'):
    scene_types = [f for f in os.listdir(scene_dir) if isdir(join(scene_dir, f)) and 'texture_library' not in f]
    scenes = dict()
    for scene in scene_types:
        objects = sorted(glob.glob("{}/*.obj".format(join(scene_dir, scene))))
        mats = sorted(glob.glob("{}/*.mtl".format(join(scene_dir, scene))))
        scenes[scene] = list(zip(objects, mats))
    object_types = [f for f in os.listdir(join(scene_dir, 'texture_library'))]
    items = dict()
    for item in object_types:
        textures = sorted(glob.glob("{}/*.jpg".format(join(scene_dir, 'texture_library', item))))
        items[item] = textures
    return scenes, items


def main():
    bpy.context.scene.render.engine = 'CYCLES'

    bpy.context.scene.camera.location.z = 1.8
    bpy.context.scene.camera.location.x = 0
    bpy.context.scene.camera.location.y = 0

    generator = SceneGenerator()
    generator.random_render()

if __name__ == "__main__":
    main()
