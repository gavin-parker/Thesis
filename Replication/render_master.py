from subprocess import Popen, PIPE
import os

"""Sends commands to the render worker processes"""
class Master:
    worker = None

    def __init__(self, blender_path):
        self.blender_path = blender_path

    def start_worker(self, scene, envmap, name):
        myargs = [
            self.blender_path,
            scene,
            "--background",
            "--python",
            "render_worker.py"
        ]
        self.worker = Popen(myargs, stdin=PIPE)
        self.start_render(envmap, name)

    def start_render(self, envmap, name):
        self.worker.communicate("{} {}".format(envmap, name))


