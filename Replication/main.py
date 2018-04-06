from models import reflectance, dematerial, stereo
import harness
import sys


def main():
    model = None
    if '--reflectance' in sys.argv:
        model = reflectance.Model()
    if '--dematerial' in sys.argv:
        model = dematerial.Model()
    if '--stereo' in sys.argv:
        model = stereo.Model()
    harness.train(model)


if __name__ == "__main__":
    main()
