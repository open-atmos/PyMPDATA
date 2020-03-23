class Options:
    def __init__(self, *,
                 n_iters: int = 2,
                 infinite_gauge: bool = False,
                 divergent_flow: bool = False,
                 flux_corrected_transport: bool = False,
                 third_order_terms: bool = False,
                 epsilon: float = 1e-15,
                 mu_coeff: float = 0
                 ):
        self._n_iters = n_iters
        self._infinite_gauge = infinite_gauge
        self._epsilon = epsilon
        self._divergent_flow = divergent_flow
        self._flux_corrected_transport = flux_corrected_transport
        self._third_order_terms = third_order_terms
        self._mu_coeff = mu_coeff

    @property
    def n_iters(self):
        return self._n_iters

    @property
    def infinite_gauge(self):
        return self._infinite_gauge

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def divergent_flow(self):
        return self._divergent_flow

    @property
    def flux_corrected_transport(self):
        return self._flux_corrected_transport

    @property
    def third_order_terms(self):
        return self._third_order_terms

    @property
    def mu_coeff(self):
        return self._mu_coeff
