#!/usr/bin/env python3

from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.algorithms import Algorithm


class GradientAllReduceAlgorithm(Algorithm):
    def __init__(self, hierarchical: bool = False, average: bool = True):
        """
        Create an instance of the
        `GradientAllReduce <https://baguasys.github.io/tutorials/algorithms/gradient-allreduce.html>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            average (bool): If True, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        self.hierarchical = hierarchical
        self.average = average

    def init_operations(
        self,
        bagua_module: BaguaModule,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        if self.hierarchical:
            bucket.append_centralized_synchronous_op(
                hierarchical=self.hierarchical,
                average=self.average,
            )
        else:
            bucket.append_centralized_synchronous_op(
                hierarchical=self.hierarchical,
                average=self.average,
            )
