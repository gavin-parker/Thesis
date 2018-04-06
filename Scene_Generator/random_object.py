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


class SceneGenerator:
    scene = bpy.context.scene
    envmap_camera = bpy.data.objects["Envmap_Camera"]
    render_camera = bpy.data.objects["Camera"]
    camera_limits = [(0.8, 1.3), (-0.1, 0.1), (-3.14, 3.14)]
    envmaps = glob.glob('{}/hdris/*.hdr'.format(scene_dir))
    tables = [obj.name for obj in scene.objects if "Table" in obj.name]
    imported_objects = []
    table = {}
    nodes = bpy.data.scenes[0].node_tree.nodes
    car = None

    def __init__(self):
        self.models = find_scene_data()
        self.material = bpy.data.materials['Mix']
        self.normal_material = bpy.data.materials['Normals']
        self.scene.use_nodes = False

    def random_material(self, objects):
        mat_prop(self.material, 'Base Color', random_colour())
        mat_prop(self.material, 'Subsurface', random.uniform(0, 0.2))
        mat_prop(self.material, 'Subsurface Color', random_colour())
        mat_prop(self.material, 'Metallic', random.uniform(0, 1))
        mat_prop(self.material, 'Specular', random.uniform(0, 1))
        mat_prop(self.material, 'Roughness', random.uniform(0, 1))
        for obj in objects:
            if obj.type != 'CAMERA' and obj.type != 'LAMP':
                if obj.data.materials:
                    obj.data.materials[0] = self.material
                else:
                    obj.data.materials.append(self.material)

                obj.active_material = self.material
                bpy.context.scene.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.object.material_slot_assign()
                bpy.ops.object.mode_set(mode='OBJECT')
                obj.select = True

    def surface_normals(self):
        for obj in self.scene.objects:
            if obj.type == 'MESH':
                # obj.data.materials.append(self.normal_material)
                bpy.context.scene.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')
                obj.data.materials.clear()
                obj.data.materials.append(self.normal_material)
                bpy.context.object.active_material_index = 0
                bpy.ops.object.material_slot_assign()
                # obj.active_material = self.normal_material
                bpy.ops.object.mode_set(mode='OBJECT')
                # bpy.context.scene.objects.active = obj
                # bpy.ops.object.mode_set(mode='EDIT')
                # bpy.context.object.active_material_index = 1
                # bpy.ops.object.material_slot_assign()
                # bpy.context.object.active_material_index = 0
                # bpy.ops.object.mode_set(mode='OBJECT')
                obj.select = True

    def place_random_object(self):
        path = random.choice(self.models)
        bpy.ops.import_scene.obj(filepath=path)
        for material in bpy.data.materials:
            if material != self.material and material != self.normal_material:
                bpy.data.materials.remove(material)
        objects = bpy.context.selected_objects[:]
        bpy.ops.transform.resize(value=(0.1, 0.1, 0.1), constraint_axis=(False, False, False),
                                 constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',
                                 proportional_edit_falloff='SMOOTH', proportional_size=1)
        # bpy.ops.mesh.uv_texture_remove()
        self.car = objects
        return path

    def clear_objects(self, objects):
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.select = True
        bpy.ops.object.delete()

    def random_render(self, name='test'):
        self.scene.camera = self.render_camera
        bpy.data.cameras[self.scene.camera.name].clip_start = 0.0001
        bpy.data.cameras[self.scene.camera.name].clip_end = 1000
        random_rotation(self.scene.camera, self.camera_limits)
        self.random_material(self.car)
        bpy.ops.view3d.camera_to_view_selected()
        envmap_id = self.light_scene()
        self.scene.view_settings.view_transform = 'Default'
        self.scene.render.image_settings.file_format = 'PNG'
        self.scene.render.resolution_percentage = 100
        self.scene.render.resolution_x = 512
        self.scene.render.resolution_y = 512
        self.nodes["File Output"].base_path = "{}/renders/normals/".format(scene_dir)
        self.nodes["File Output"].file_slots[0].path = name + '_'
        bpy.data.scenes['Scene'].render.filepath = "{}/renders/left/{}".format(scene_dir,
                                                                               '{}.png'.format(name))
        bpy.ops.render.render(write_still=True)
        move_object(self.scene.camera, (0.5, 0.0, 0.0))
        bpy.data.scenes['Scene'].render.filepath = "{}/renders/right/{}".format(scene_dir,
                                                                                '{}.png'.format(name))


        self.scene.use_nodes = False
        bpy.ops.render.render(write_still=True)
        self.scene.use_nodes = False
        self.surface_normals()
        bpy.context.scene.render.layers["RenderLayer"].use_sky = False
        bpy.data.scenes['Scene'].render.filepath = "{}/renders/norms/{}".format(scene_dir,
                                                                         '{}.png'.format(name))

        bpy.ops.render.render(write_still=True)
        bpy.context.scene.render.layers["RenderLayer"].use_sky = True
    def light_scene(self):
        envmap = random.choice(self.envmaps)
        bpy.data.images.load(envmap, check_existing=False)
        self.scene.world.use_nodes = True
        self.scene.world.node_tree.nodes['Environment Texture'].image = bpy.data.images[os.path.basename(envmap)]
        return os.path.basename(envmap).split('.')[0]

    def render_envmap(self, name='test'):
        for obj in self.car:
            obj.hide_render = True
        self.scene.render.resolution_x = 32
        self.scene.render.resolution_y = 32
        self.scene.render.resolution_percentage = 100
        self.envmap_camera.rotation_euler = self.render_camera.rotation_euler
        self.scene.camera = self.envmap_camera
        self.scene.view_settings.view_transform = 'Raw'
        self.scene.render.image_settings.file_format = 'HDR'
        bpy.data.scenes['Scene'].render.filepath = "{}/renders/envmaps/{}".format(scene_dir, '{}.hdr'.format(name))
        bpy.ops.render.render(write_still=True)
        for obj in self.car:
            obj.hide_render = False


def mat_prop(mat, property, val):
    mat.node_tree.nodes["Principled BSDF"].inputs[property].default_value = val


def random_colour():
    return (random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), 1)


def move_object(object, vector):
    rightvec = mathutils.Vector(vector)
    inv = object.matrix_world.copy()
    inv.invert()
    vec_rot = rightvec * inv
    object.location = object.location + vec_rot


def random_rotation(object, limits):
    object.rotation_euler.x = random.uniform(limits[0][0], limits[0][1])
    object.rotation_euler.y = random.uniform(limits[1][0], limits[1][1])
    object.rotation_euler.z = random.uniform(limits[2][0], limits[2][1])


def find_scene_data():
    categories = dict()
    models = glob.glob('{}/models/*/*/*.obj'.format(scene_dir))
    return models


def extract_material(category, materials, limit=4):
    return [i[0] for i in process.extract(category, materials, limit=limit)]


def main():
    prefix = 0
    for arg in sys.argv:
        if 'prefix' in arg:
            prefix = int(arg.split('=')[1])
    bpy.context.scene.render.engine = 'CYCLES'
    try:
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.scene.render.tile_x = 512
        bpy.context.scene.render.tile_y = 512
    except TypeError:
        bpy.context.scene.render.tile_x = 64
        bpy.context.scene.render.tile_y = 64
        pass
    # bpy.context.user_preferences.addons['cycles'].preferences.devices[1].use = True
    prefs = bpy.context.user_preferences.addons['cycles'].preferences
    # bpy.ops.wm.addon_enable(module='materials_utils')
    print(prefs.compute_device_type)
    generator = SceneGenerator()
    filename = generator.place_random_object()
    filename = os.path.dirname(filename).split('/')[-1]

    for i in range(prefix, prefix + 10):
        generator.random_render(str(i))
        generator.render_envmap(str(i))
    generator.clear_objects(generator.car)


if __name__ == "__main__":
    main()
