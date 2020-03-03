def convergence_in_space(nr: list, results):
    err_data = {}
    for coord in results.keys():
        err_data[coord] = {}
        for opts in results[coord]["error_L2"].keys():
            err_data[coord][opts] = []

    for nri, nr in enumerate(nrs):
        data, setup = figure_data(nr=nr)
        for coord in data.keys():
            for opts in data[coord]["error_L2"].keys():
                err_data[coord][opts].append(data[coord]["error_L2"][opts])
    return err_data


def convergence_in_time(dts: list, results):
    err_data = {}
    for coord in results.keys():
        err_data[coord] = {}
        for opts in results[coord]["error_L2"].keys():
            err_data[coord][opts] = []

    for dti, dt in enumerate(nrs):
        data, setup = figure_data(nr=nr)
        for coord in data.keys():
            for opts in data[coord]["error_L2"].keys():
                err_data[coord][opts].append(data[coord]["error_L2"][opts])
    return err_data