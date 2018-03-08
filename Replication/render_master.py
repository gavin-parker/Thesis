import subprocess32
import os

"""Sends commands to the render worker processes"""
class Master:
    worker = None
    pipe = open(os.devnull, 'w')
    def __init__(self, blender_path, silent=True):
        self.blender_path = blender_path
        if not silent:
            self.pipe = None

    def start_worker(self, scene, envmap, name, render_background=False):
        r_b = "none"
        if render_background:
            r_b = "render_background"

        myargs = [
            self.blender_path,
            scene,
            "--background",
            "--python",
            "render_worker.py",
            envmap,
            name,
            r_b
        ]
        try:
            subprocess32.call(myargs, shell=False, timeout=30, stdout=self.pipe)
        except subprocess32.TimeoutExpired:
            pass