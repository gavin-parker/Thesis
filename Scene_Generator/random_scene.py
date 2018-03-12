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
import materials_cycles_converter

class SceneGenerator:
    scene = bpy.context.scene
    envmap_camera = bpy.data.objects["Envmap_Camera"]
    render_camera = bpy.data.objects["Camera"]
    camera_limits = [(0.8, 1.1), (-0.1, 0.1), (-3.14, 3.14)]
    envmaps = glob.glob('/home/gavin/hdris/*.hdr')
    tables = [obj.name for obj in scene.objects if "Table" in obj.name]
    imported_objects = []
    table = {}
    output_path = '/mnt/black/scene_data/'
    def __init__(self):
        self.materials, self.models = find_scene_data()
        print(self.tables)

    def randomize_table(self):
        for table in self.tables:
            bpy.data.objects[table].hide_render = True
            bpy.data.objects[table].select = False
        table = random.choice(self.tables)
        bpy.data.objects[table].hide_render = False
        bpy.data.objects[table].select = True
        mat = random.choice(self.materials['table'])
        self.scene.objects[table].active_material = mat
        self.table = bpy.data.objects[table]
        bpy.ops.view3d.camera_to_view_selected()
        return

    def place_random_object(self, bounds):
        bpy.ops.import_scene.obj(filepath=random.choice(self.models))
        objects = bpy.context.selected_objects[:]
        lowest_pt = 100
        position_x = random.uniform(bounds[0][0], bounds[0][1])
        position_y = random.uniform(bounds[1][0], bounds[1][1])
        for object in objects:
            lowest_pt = min(lowest_pt, (min([(object.matrix_world * v.co).z for v in object.data.vertices])))
        for object in objects:
            object.location.z = object.location.z - lowest_pt + 0.1
            object.location.x = object.location.x + position_x
            object.location.y = object.location.y + position_y
            object.rotation_euler.z = random.uniform(-3.14,3.14)
            #mat = random.choice(self.materials['metal'])
            #object.active_material = mat
            object.select = False
        return objects

    def place_random_objects(self):
        count = random.randint(1,4)
        bounds = [(-1.5,1.5), (-1.5,1.5)]
        self.imported_objects = []
        for i in range(count):
            objects = self.place_random_object(bounds)
            self.imported_objects.extend(objects)
        materials_cycles_converter.mlrefresh(bpy.context)

    def clear_objects(self):
        bpy.ops.object.select_all(action='DESELECT')
        for obj in self.imported_objects:
            obj.select = True
        bpy.ops.object.delete()

    def random_render(self, name='test'):
        self.scene.camera = self.render_camera
        random_rotation(self.scene.camera, self.camera_limits)
        self.randomize_table()
        self.place_random_objects()
        self.light_scene()
        self.scene.view_settings.view_transform = 'Default'
        self.scene.render.image_settings.file_format = 'PNG'
        bpy.data.scenes['Scene'].render.filepath = "{}/renders/{}".format(self.output_path,'{}.png'.format(name))
        bpy.ops.render.render(write_still=True)
        move_object_right(self.scene.camera)
        bpy.data.scenes['Scene'].render.filepath = "{}/renders/{}".format(self.output_path,'{}_b.png'.format(name) )
        bpy.ops.render.render(write_still=True)
        self.clear_objects()

    def light_scene(self):
        envmap = random.choice(self.envmaps)
        bpy.data.images.load(envmap, check_existing=False)
        self.scene.world.use_nodes = True
        self.scene.world.node_tree.nodes['Environment Texture'].image = bpy.data.images[os.path.basename(envmap)]

    def render_envmap(self, name='test'):
        self.envmap_camera.rotation_euler = self.render_camera.rotation_euler
        self.scene.camera = self.envmap_camera
        self.scene.view_settings.view_transform = 'Raw'
        self.scene.render.image_settings.file_format = 'HDR'
        bpy.data.scenes['Scene'].render.filepath = "{}/envmaps/{}".format(self.output_path,'{}.hdr'.format(name))
        bpy.ops.render.render(write_still=True)


def move_object_right(object):
    rightvec = mathutils.Vector((0.1, 0.0, 0.0))
    inv = object.matrix_world.copy()
    inv.invert()
    vec_rot = rightvec * inv
    object.location = object.location + vec_rot


def random_rotation(object, limits):
    object.rotation_euler.x = random.uniform(limits[0][0], limits[0][1])
    object.rotation_euler.y = random.uniform(limits[1][0], limits[1][1])
    object.rotation_euler.z = random.uniform(limits[2][0], limits[2][1])


def find_scene_data(scene_dir='/mnt/black/scene_data'):
    material_file = '{}/materials.blend'.format(scene_dir)
    with bpy.data.libraries.load(material_file, link=False) as (data_from, data_to):
        data_to.materials = [name for name in data_from.materials]

    models = glob.glob('{}/models/*/*/*.obj'.format(scene_dir))
    categories = dict()
    categories['wood'] = extract_material('wood', data_to.materials)
    categories['metal'] = extract_material('metal', data_to.materials)
    categories['plastic'] = extract_material('plastic', data_to.materials)
    categories['table'] = categories['wood'] + categories['plastic']
    return categories, models


def extract_material(category, materials):
    return [i[0] for i in process.extract(category, materials, limit=5)]


def main():
    bpy.context.scene.render.engine = 'CYCLES'

    generator = SceneGenerator()
    for i in range(10):
        generator.random_render(str(i))
        generator.render_envmap(str(i))


if __name__ == "__main__":
    main()
