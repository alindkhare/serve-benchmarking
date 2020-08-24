import git
import os

ROOT_DIR = git.Repo('.', search_parent_directories=True).working_tree_dir
RESULT_DIR = os.path.join(ROOT_DIR, "profiles/results")

label_map = {
    "bytes_tensors": "pickled",
    "direct_tensors": "unpickled"
}
tensor_types = ["torch", "np", "tf"]
