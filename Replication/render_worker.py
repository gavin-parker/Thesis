import bpy

"""Class to handle rendering of test scenes in Blender via subprocess"""
class RenderWorker:
    def __init__(self):
        bpy.context.scene.render.engine = 'CYCLES'

    def render_image(self, name):
        bpy.data.scenes['Scene'].render.filepath = name
        bpy.ops.render.render(write_still=True)



def main():
    worker = RenderWorker()
    worker.render_image('test.png')

if __name__ == "__main__":
    main()
