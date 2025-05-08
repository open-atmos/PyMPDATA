from PyMPDATA import Options

OPTIONS = {
    "UPWIND": Options(
        n_iters=1,
        non_zero_mu_coeff=True,
    ),
    "MPDATA": Options(
        n_iters=3,
        nonoscillatory=True,
        non_zero_mu_coeff=True,
    ),
}
