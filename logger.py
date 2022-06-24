import sys
from tqdm import tqdm
from typing import Iterable, Optional
from _io import TextIOWrapper


class Logger:
    def __init__(self):
        self.default_out = sys.stdout

    def log(self, message: object, file: Optional[TextIOWrapper] = None):
        print(message, file=self.default_out if file is None else file)

    def get_tqdm(self, iter_obj: Iterable, desc: str, leave=True):
        return tqdm(iter_obj, desc=desc.ljust(30), leave=leave, file=self.default_out, ncols=150)


logger = Logger()
