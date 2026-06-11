from typing import Any, Callable, TypeVar

import numpy as np
import networkx as nx
from numpy import pi

T = TypeVar("T")
R = TypeVar("R")


def memoize_last(fn: Callable[[T], R]) -> Callable[[T], R]:
    """Memoize a unary call keyed on its argument's identity.

    Returns a wrapper that caches the most recent (arg, result) pair and
    re-runs ``fn`` only when called with a different argument (by ``is``).
    Used to share per-Y state across repeated calls at the same base
    point, e.g. the inner CG of RTR.
    """
    cache: list[Any] = [None, None]

    def wrapped(arg: T) -> R:
        if cache[0] is not arg:
            cache[:] = [arg, fn(arg)]
        return cache[1]

    return wrapped


def level2_descendants(G: nx.DiGraph, node_id):
    successors = G.successors(node_id)

    desc = []
    for su in successors:
        desc += [G.successors(su)]

    return flatten(desc)


def wraptopi(e):
    return np.mod(e + pi, 2 * pi) - pi


def flatten(l: list) -> list:
    return [item for sublist in l for item in sublist]


def list_to_variable_dict(l: list, label="p", index_start=1):
    if type(l) is dict:
        return l
    var_dict = {label+f'{index_start + idx}': val for idx, val in enumerate(l)}
    return var_dict


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
