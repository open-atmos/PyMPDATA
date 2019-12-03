
def vector(opts, GCh, ih, halo):
    start = ih.start
    stop = ih.stop

    if opts["bcond"] == 'periodic':
        GCh[:start] = GCh[stop - (halo-1):stop]
        GCh[stop:] = GCh[start:start + (halo-1)]
    else:
        GCh[:start] = 0
        GCh[stop:] = 0


def scalar(opts, psi, i, halo):
    start = i.start
    stop = i.stop

    if opts["bcond"] == 'periodic':
        psi[:start] = psi[stop - halo: stop]
        psi[stop:] = psi[start:start + halo]
    else:
        psi[:start] = 0
        psi[stop:] = 0
