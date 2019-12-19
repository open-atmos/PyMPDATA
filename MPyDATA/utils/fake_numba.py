# https://github.com/ptooley/numbasub/blob/master/src/numbasub/nonumba.py

import functools


class FakeType:
    def __call__(self, *args, **kwargs):
        pass

    def __getitem__(self, item):
        pass


int64 = FakeType()
float64 = FakeType()
boolean = FakeType()


def optional_arg_decorator(fn):
    @functools.wraps(fn)
    def wrapped_decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return fn(args[0])
        else:
            def real_decorator(decoratee):
                return fn(decoratee, *args, **kwargs)
            return real_decorator
    return wrapped_decorator


@optional_arg_decorator
def jit(func, *args, **kwargs):
    return func


@optional_arg_decorator
def njit(func, *args, **kwargs):
    return func


@optional_arg_decorator
def jitclass(func, *args, **kwargs):
    return func
