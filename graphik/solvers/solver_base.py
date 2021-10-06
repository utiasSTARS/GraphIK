import numpy as np
import sympy as sp
from abc import ABC, abstractmethod
from graphik.graphs.graph_base import ProblemGraph

class GraphProblemSolver(ABC):
    def __init__(self, params):
        self.params = params  # Uses the abstract setter
        self._cost = None

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, cost):
        self._cost = cost

    @abstractmethod
    def solve(self, graph: ProblemGraph, problem_params: dict = None):
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @params.setter
    @abstractmethod
    def params(self, params):
        pass


