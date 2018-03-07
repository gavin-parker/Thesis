import bpy
import sys
import os
import random
"""Class to handle rendering of test scenes in Blender via subprocess"""


class RenderWorker:
    def __init__(self):
        bpy.context.scene.render.engine = 'CYCLES'

    def render_image(self, name, envmap, render_bg):
        print('{}/envmaps/{}'.format(os.getcwd(), envmap))
        self.randomize_material()
        bpy.data.objects['Sphere'].hide_render = False

        """Render the sphere with lighting"""
        bpy.data.images.load('{}/envmaps/{}'.format(os.getcwd(), envmap), check_existing=False)
        bpy.data.scenes['Scene'].world.node_tree.nodes['Environment Texture'].image = bpy.data.images[envmap]
        bpy.data.scenes['Scene'].render.filepath = "renders/{}".format(name)
        bpy.data.scenes['Scene'].world.cycles_visibility.camera = False
        bpy.ops.render.render(write_still=True)
        """Render the Background"""
        if render_bg:
            bpy.data.objects['Sphere'].hide_render = True
            bpy.data.scenes['Scene'].world.cycles_visibility.camera = True
            bpy.data.scenes['Scene'].render.filepath = "renders/bg_{}".format(name)
            bpy.ops.render.render(write_still=True)

    def randomize_material(self):
        x = random.randint(0,1)
        col = (random.random(), random.random(), random.random(), 1)
        bpy.data.objects['Sphere'].active_material_index = x
        mat = ["Diffuse BSDF", "Glossy BSDF"][x]
        if x:
            bpy.data.objects['Sphere'].active_material.node_tree.nodes[mat].inputs[0].default_value = col
        else:
            bpy.data.objects['Sphere'].active_material.node_tree.nodes[mat].inputs[0].default_value = col
        bpy.ops.object.editmode_toggle()
        bpy.ops.object.material_slot_assign()

def main():
    worker = RenderWorker()
    print("starting render")
    idx = [i for i, s in enumerate(sys.argv) if '.hdr' in s][0]
    command = sys.argv[idx:-1]
    print(command)
    envmap = command[0]
    name = command[1]
    render_bg = "render_background" in command
    worker.render_image(name, envmap, render_bg)


if __name__ == "__main__":
    main()
