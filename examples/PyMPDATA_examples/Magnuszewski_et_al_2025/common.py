from PyMPDATA import Options

OPTIONS = {
    "UPWIND": Options(
        n_iters=1,
        non_zero_mu_coeff=True,
    ),
    "MPDATA (2 it.)": Options(
        n_iters=2,
        nonoscillatory=True,
        non_zero_mu_coeff=True,
    ),
    "MPDATA (4 it.)": Options(
        n_iters=4,
        nonoscillatory=True,
        non_zero_mu_coeff=True,
    ),
}
