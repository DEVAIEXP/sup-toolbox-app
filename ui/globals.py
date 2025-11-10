import threading


# These variables live at the module level.
# When multiple workers import this module, they will all reference
# the same objects in the parent process's memory (due to how Python forks processes on Linux).
sup_toolbox_pipe = None
pipeline_lock = threading.Lock()
