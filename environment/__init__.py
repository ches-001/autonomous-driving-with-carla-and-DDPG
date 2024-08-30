# The navigation and tools submodules of this module are provided by the Python client API for carla
# which is stored PythonAPI sub-directory of the main carla directory.
import glob
import os
import sys
try:
    inst_dir = "CARLA_0.9.8/WindowsNoEditor" if os.name == "nt" else "CARLA_0.9.8"
    path = glob.glob(os.path.join(inst_dir, "PythonAPI/carla/dist/carla-*%d.%d-%s.egg" % (
        sys.version_info.major,
        sys.version_info.minor,
        "win-amd64" if os.name == "nt" else "linux-x86_64")))[0]
    
    sys.path.append(path)
except IndexError:
    pass