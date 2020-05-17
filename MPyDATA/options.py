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
                 non_zero_mu_coeff: bool = False,
                 trapez: bool = False
                 ):
        self._values = {'n_iters': n_iters, 'infinite_gauge': infinite_gauge, 'epsilon': epsilon,
                        'divergent_flow': divergent_flow, 'flux_corrected_transport': flux_corrected_transport,
                        'third_order_terms': third_order_terms, 'non_zero_mu_coeff': non_zero_mu_coeff,
                        'trapez': trapez}

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
    def non_zero_mu_coeff(self):
        return self._values['non_zero_mu_coeff']

    def __hash__(self):
        value = hash(str(self._values))
        return value

    def __eq__(self, other):
        return other.__hash__() == self.__hash__()

    @property
    def n_halo(self):
        if self.divergent_flow or self.flux_corrected_transport or self.third_order_terms:
            return 2
        else:
            return 1

    @property
    def trapez(self):
        return self._values['trapez']

    @property
    def jit_flags(self):
        return {
            "fastmath": True,
            "error_model": 'numpy',
            "boundscheck": False,
        }
