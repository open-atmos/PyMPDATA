from .meta import make_meta


class Field:
    def __init__(self, *, grid):
        self.grid = grid
        self.meta = make_meta(False, grid)
