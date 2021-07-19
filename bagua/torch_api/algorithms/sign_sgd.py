from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from typing import List
import torch


class Sign_sgd:
    """
    This is the base class that all Bagua algorithms inherit.

    It provides methods that can be override to implement different kinds of
    distributed algorithms.
    """

    def need_reset(self) -> bool:
        """
        Returns:
            True if all initialization methods of the current algorithms should
            be called again. This is useful for algorithms that has multiple
            stages where each stage needs different initializations.
        """
        return False

    def init_tensors(self, bagua_module: BaguaModule) -> List[BaguaTensor]:
        """
        Given a BaguaModule, return Bagua tensors to be used in Bagua for later
        operations.

        Args:
            bagua_module: A PyTorch module initialized by
                ``with_bagua(...)`` method.

        Returns:
            A list of Bagua tensors for communication.
        """

        parameters = bagua_module.bagua_build_params()
        tensors = []
        for name, param in parameters.__reversed__():
            grad = param.bagua_ensure_grad().ensure_bagua_tensor(
                name, bagua_module.bagua_module_name
            )
            param._bagua_grad = grad
            tensors.append(grad)
        self._communication_tensor_names = set(name for name, _ in parameters)
        assert len(self._communication_tensor_names) == len(
            tensors
        ), "tensor names should be unique"
        return tensors

    def tensors_to_buckets(self, tensors: List[List[BaguaTensor]]) -> List[BaguaBucket]:
        """
        Given the bucketing suggestion from Bagua, return the actual Bagua buckets.
        The default implementation follows the suggestion to do the bucketing.

        Args:
            tensors: Bagua tensors grouped in different
                lists, representing Bagua's suggestion on how to bucketing the
                tensors.

        Returns:
            A list of Bagua buckets.
        """
        bagua_buckets = []
        for idx, bucket in enumerate(tensors):
            bagua_bucket = BaguaBucket(
                bucket, flatten=True, name=str(idx)
            )  # TODO: check duplicated names
            bagua_buckets.append(bagua_bucket)
        return bagua_buckets

    def init_forward_pre_hook(self, bagua_module: BaguaModule):
        """Given a `BaguaModule`, return a hook function that will be executed before the
        forward process.

        Args:
            bagua_module: A PyTorch module initialized by
                ``with_bagua(...)`` method.

        Returns:
            A function that takes the model's input.
        """

        def hook(input):
            pass

        return hook

    def init_backward_hook(self, bagua_module: BaguaModule):
        """Given a `BaguaModule`, return a hook function that will be executed on every
        parameter's gradient computation completion.

        Args:
            bagua_module: A PyTorch module initialized by
                ``with_bagua(...)`` method.

        Returns:
            A function that takes the name of a parameter (as in
            torch.nn.Module.named_parameters()) and the parameter itself.
        """

        def hook(parameter_name, parameter):
            if parameter_name in self._communication_tensor_names:
                assert (
                    parameter._bagua_grad.data_ptr() == parameter.grad.data_ptr()
                ), "bagua grad data_ptr should match parameter grad"
                parameter._bagua_grad.bagua_mark_communication_ready()

        return hook

    def init_post_backward_hook(self, bagua_module: BaguaModule):
        """Given a `BaguaModule`, return a hook function that will be executed when the
        backward pass is done.

        Args:
            bagua_module: A PyTorch module initialized by
                ``with_bagua(...)`` method.

        Returns:
            A function that takes no argument.
        """

        def hook():
            bagua_module._bagua_backend.wait_pending_comm_ops()

        return hook

    def init_post_optimizer_step_hook(self, bagua_module: BaguaModule):
        """Given a `BaguaModule`, return a hook function that will be executed when the
        ``optimizer.step()`` is done.

        Args:
            bagua_module: A PyTorch module initialized by
                ``with_bagua(...)`` method.

        Returns:
            A function that takes the optimizer that is called step().
        """

        def hook(optimizer: torch.optim.Optimizer):
            pass

        return hook

    def init_operations(
        self,
        bagua_module: BaguaModule,
        bucket: BaguaBucket,
    ):
        """Given a `BaguaModule`, and a Bagua bucket, register operations to be
        executed on the bucket.

        Args:
            bagua_module: A PyTorch module initialized by
                ``with_bagua(...)`` method.
            bucket: A single bucket to register operations.
        """
