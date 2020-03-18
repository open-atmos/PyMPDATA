class Options:
    def __init__(self,
                 n_iters: int = 2,
                 infinite_gauge: bool = False,
                 epsilon: float = 1e-15
                 ):
        self._n_iters = n_iters
        self._infinite_gauge = infinite_gauge
        self._epsilon = epsilon

    @property
    def n_iters(self):
        return self._n_iters

    @property
    def infinite_gauge(self):
        return self._infinite_gauge

    @property
    def epsilon(self):
        return self._epsilon