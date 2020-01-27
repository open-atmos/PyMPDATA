"""
Created at 21.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""


class EulerianFields:
    def __init__(self, mpdatas: dict):
        self.mpdatas = mpdatas

    def step(self, n_iters: int, debug=False):
        for mpdata in self.mpdatas.values():
            mpdata.step(n_iters, debug=debug)
