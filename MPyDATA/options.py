"""
Created at 03.2020
"""


class Options:
    def __init__(self, *,
                 n_iters: int = 2,
                 infinite_gauge: bool = False,
                 divergent_flow: bool = False,
                 flux_corrected_transport: bool = False,
                 third_order_terms: bool = False,
                 epsilon: float = 1e-15,
                 mu_coeff: float = 0,
                 ):
        self._values = {'n_iters': n_iters, 'infinite_gauge': infinite_gauge, 'epsilon': epsilon,
                        'divergent_flow': divergent_flow, 'flux_corrected_transport': flux_corrected_transport,
                        'third_order_terms': third_order_terms, 'mu_coeff': mu_coeff}

        if flux_corrected_transport and n_iters < 2:
            raise ValueError()

    @property
    def n_iters(self):
        return self._values['n_iters']

    @property
    def infinite_gauge(self):
        return self._values['infinite_gauge']

    @property
    def epsilon(self):
        return self._values['epsilon']

    @property
    def divergent_flow(self):
        return self._values['divergent_flow']

    @property
    def flux_corrected_transport(self):
        return self._values['flux_corrected_transport']

    @property
    def third_order_terms(self):
        return self._values['third_order_terms']

    @property
    def mu_coeff(self):
        return self._values['mu_coeff']

    def __str__(self):
        return str(self._values)
