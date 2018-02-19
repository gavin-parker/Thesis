from subprocess import Popen, PIPE
import os

"""Sends commands to the render worker processes"""
class Master:
    worker = None

    def __init__(self, blender_path):
        self.blender_path = blender_path

    def start_worker(self, scene):
        myargs = [
            self.blender_path,
            scene,
            "--background",
            "--python",
            "render_worker.py"
        ]
        self.worker = Popen(myargs, stdin=PIPE)

    def start_render(self, envmap):
        self.worker.stdin.write("{}\n".format(envmap))

    def stop_worker(self):
        self.worker.stdin.write('stop\n')


def test():
    master = Master('/home/gavin/blender-2.79-linux-glibc219-x86_64/blender')
    master.start_worker('test_elephant.blend')
    for filename in os.listdir('envmaps'):
        print(filename)
        master.start_render(filename)
    master.stop_worker()

if __name__ == "__main__":
    test()
