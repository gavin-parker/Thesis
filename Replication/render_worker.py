import bpy
import sys
import os

"""Class to handle rendering of test scenes in Blender via subprocess"""


class RenderWorker:
    def __init__(self):
        bpy.context.scene.render.engine = 'CYCLES'

    def render_image(self, name, envmap):
        print('{}/{}'.format(os.getcwd(),envmap))
        bpy.data.images.load('{}/{}'.format(os.getcwd(),envmap), check_existing=False)
        bpy.data.scenes['Scene'].world.node_tree.nodes['Environment Texture'].image = bpy.data.images[envmap]
        bpy.data.scenes['Scene'].render.filepath = name
        bpy.ops.render.render(write_still=True)


def main():
    worker = RenderWorker()
    for command in sys.stdin:
        if '.hdr' in command:
            commands = command.split()
            envmap = commands[0]
            name = commands[1]
            worker.render_image(name, envmap)
            return


if __name__ == "__main__":
    main()
