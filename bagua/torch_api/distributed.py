from __future__ import annotations
from bagua.torch_api.communication import get_backend
import bagua
from bagua.torch_api.utils import to_bagua_datatype, StatisticalAverage
from bagua.torch_api.env import get_autotune_level, get_rank
from bagua.bagua_define import (
    TensorDeclaration,
    BaguaHyperparameter,
)
import gorilla
import time
import logging
import torch
import torch.nn
import itertools
from typing import List, Tuple


@gorilla.patches(torch.nn.Module, filter=lambda name, obj: "bagua" in name)
class BaguaModule:
    """
    This class patches `torch.nn.Module` with several methods to enable Bagua
    functionalities.

    :ivar bagua_optimizers: The optimizers passed in by ``with_bagua(...)``.
    :vartype bagua_optimizers: List[torch.optim.Optimizer]

    :ivar bagua_algorithm: The algorithm passed in by ``with_bagua(...)``.
    :vartype bagua_algorithm: bagua.torch_api.algorithms.Algorithm

    :ivar parameters_to_ignore: The parameter names in ``"{module_name}.{param_name}"`` format to ignore
        when calling ``self.bagua_build_params()``.
    :vartype parameters_to_ignore: List[str]

    :ivar bagua_train_step_counter: Number of iterations in training mode
    :vartype bagua_train_step_counter: int

    :ivar bagua_buckets: All Bagua buckets in a list.
    :vartype bagua_buckets: List[bagua.torch_api.bucket.BaguaBucket]
    """

    __id_iter = itertools.count()

    def bagua_build_params(self) -> List[Tuple[str, torch.nn.Parameter]]:
        """
        Build tuple of ``(parameter_name, parameter)`` for all parameters that
        require grads and not in the ``_bagua_params_and_buffers_to_ignore`` attribute.
        """
        modules_and_parameters = [
            (module, parameter)
            for module_name, module in self.named_modules()
            for parameter in [
                (f"{module_name}.{param_name}", param)
                # Note that we access module.named_parameters instead of
                # parameters(module). parameters(module) is only needed in the
                # single-process multi device case, where it accesses replicated
                # parameters through _former_parameters.
                for param_name, param in module.named_parameters(recurse=False)
                if param.requires_grad
                and f"{module_name}.{param_name}" not in self.parameters_to_ignore
            ]
        ]

        # Deduplicate any parameters that might be shared across child modules.
        memo = set()
        # "p not in memo" is the deduplication check.
        # "not memo.add(p)" is always True, and it's only there to cause "add(p)" if needed.
        modules_and_parameters = [
            (m, p)
            for m, p in modules_and_parameters
            if p not in memo and not memo.add(p)
        ]

        # Build list of parameters.
        parameters = [parameter for _, parameter in modules_and_parameters]

        # Checks if a module will produce a sparse gradient.
        def produces_sparse_gradient(module):
            if isinstance(module, torch.nn.Embedding) or isinstance(
                module, torch.nn.EmbeddingBag
            ):
                return module.sparse
            return False

        # Build list of booleans indicating whether or not to expect sparse
        # gradients for the corresponding parameters.
        expect_sparse_gradient = [
            produces_sparse_gradient(module) for module, _ in modules_and_parameters
        ]

        if any(expect_sparse_gradient):
            raise NotImplementedError("sparse gradient not supported yet")

        return parameters

    def _bagua_broadcast_parameters(self):
        """
        Broadcast model and optimizer states.
        """
        from bagua.torch_api.communication import broadcast

        module_states = self.bagua_build_params()
        for name, state in module_states:
            broadcast(state, src=0)
        for optimizer in self.bagua_optimizers:
            optimizer_state_dict = optimizer.state_dict()["state"]
            for state in optimizer_state_dict.values():
                for inner_state in state.values():
                    if isinstance(
                        inner_state, torch.Tensor
                    ):  # TODO: consider the case where this is a scalar
                        broadcast(inner_state, src=0)

    def _bagua_autotune_step(self):
        CYCLE_STEP = 100
        start_time = time.time()

        if (
            self.bagua_train_step_counter != 0
            and self.bagua_train_step_counter % CYCLE_STEP == 0
        ):
            # get speed metrics
            time_since_last_update = time.time() - self._bagua_autotune_last_report_time
            speed = self._speed_metrics.get(time_since_last_update)

            # report metrics
            # TODO: @shjwudp add support for reporting tensor completion order
            # so that the autotune service does not rely on tensor registration
            # order
            rsp = self._bagua_autotune_client.report_metrics(
                model_name=self.bagua_module_name,
                rank=get_rank(),
                unix_timestamp=time.time(),
                train_iter=self.bagua_train_step_counter,
                hyperparameters=self._bagua_hyperparameters.dict(),
                speed=speed,
            )
            assert rsp.status_code == 200, "Unexpected rsp={}".format(rsp)

            # update parameters
            self._bagua_reset_algorithm_buckets()
            self._bagua_autotune_last_report_time = time.time()

        logging.info("autotune overhead=%s", time.time() - start_time)

    def with_bagua(  # pytype: disable=module-attr
        self,
        optimizers: List[torch.optim.Optimizer],
        algorithm: "bagua.torch_api.algorithms.Algorithm",
    ) -> BaguaModule:
        r"""``with_bagua`` enables easy distributed data parallel training on a
        ``torch.nn.Module``.

        Arguments:
            optimizers: Optimizer(s) used by the
                module. It can contain one or more PyTorch optimizers.
            algorithm: Distributed algorithm
                used to do the actual communication and update.

        Returns:
            The original module, with Bagua related environments initialized.

        .. note::
            If we want to ignore some layers for communication, we can first check
            these layer's corresponding keys in the module's ``state_dict`` (they are
            in ``"{module_name}.{param_name}"`` format), then assign the list of
            keys to ``your_module._bagua_params_and_buffers_to_ignore``.

        Examples::

            >>> model = torch.nn.Sequential(
            ...      torch.nn.Linear(D_in, H),
            ...      torch.nn.ReLU(),
            ...      torch.nn.Linear(H, D_out),
            ...    )
            >>> optimizer = torch.optim.SGD(
            ...      model.parameters(),
            ...      lr=0.01,
            ...      momentum=0.9
            ...    )
            >>> model = model.with_bagua(
            ...      [optimizer],
            ...      GradientAllReduce()
            ...    )
        """

        self.bagua_module_name = "{}_{}".format(
            self.__class__.__name__, next(BaguaModule.__id_iter)
        )

        self.bagua_optimizers = optimizers
        self.bagua_algorithm = algorithm
        self.parameters_to_ignore = (
            []
        )  #: the parameter names to ignore during communication
        if hasattr(self, "_bagua_params_and_buffers_to_ignore"):
            self.parameters_to_ignore.extend(self._bagua_params_and_buffers_to_ignore)
        if hasattr(
            self, "_ddp_params_and_buffers_to_ignore"
        ):  # for compatibility with PyTorch DDP
            self.parameters_to_ignore.extend(self._ddp_params_and_buffers_to_ignore)
        self.bagua_train_step_counter = 0
        """
        Number of iterations in training mode.
        """
        self.bagua_buckets = []
        """
        All Bagua buckets in a list.
        """
        self._bagua_autotune_last_report_time = time.time()
        self._bagua_autotune_completed = False
        self._bagua_framework_hooks = (
            []
        )  # hooks for bagua framework logic, not cleared when changing algorithms
        self._bagua_algorithm_hooks = []
        self._bagua_backend = get_backend(self.bagua_module_name)
        self._bagua_hyperparameters = BaguaHyperparameter()
        self._speed_metrics_switch_on = get_autotune_level() >= 1
        self._speed_metrics = StatisticalAverage()

        def autotune_hook(self, input):
            if self.training:
                if get_autotune_level() >= 1 and not self._bagua_autotune_completed:
                    self._bagua_autotune_step()

        def clear_post_backward_callback_queued_hook(self, input):
            self._is_post_backward_callback_queued = False

        def num_iteration_step_hook(self, input):
            if self.training:
                self.bagua_train_step_counter += 1

        def algorithm_reset_hook(self, input):
            if self.bagua_algorithm.need_reset():
                self._bagua_init_algorithm()

        def algorithm_forward_pre_hook(self, input):
            if self.training:
                self.bagua_algorithm.init_forward_pre_hook(self)(input)

        def record_speed_metrics_event(self, _):
            if not self._speed_metrics_switch_on:
                return

            if hasattr(self, "_last_event_pair"):
                (start, stop) = self._last_event_pair
                try:
                    elapsed_time_s = start.elapsed_time(stop) / 1000.0
                    total_bytes = sum(bucket.bytes() for bucket in self.bagua_buckets)
                    total_gbytes = total_bytes / 1024.0 ** 3
                    speed = total_gbytes / elapsed_time_s
                    self._speed_metrics.record(speed)
                except RuntimeError as err:
                    logging.debug("Ignore cuda err={}".format(err))

            start_event = torch.cuda.Event(enable_timing=True)
            self._speed_metrics_end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.current_stream().record_event(start_event)
            self._last_event_pair = (start_event, self._speed_metrics_end_event)

        self._bagua_framework_hooks.extend(
            [
                self.register_forward_pre_hook(num_iteration_step_hook),
                self.register_forward_pre_hook(algorithm_reset_hook),
                self.register_forward_pre_hook(algorithm_forward_pre_hook),
                self.register_forward_pre_hook(record_speed_metrics_event),
                self.register_forward_pre_hook(autotune_hook),
                self.register_forward_pre_hook(
                    clear_post_backward_callback_queued_hook
                ),
            ]
        )

        # get communicators
        self._bagua_inter_node_communicator = (
            self._bagua_backend.internode_communicator
        )
        self._bagua_intra_node_communicator = (
            self._bagua_backend.intranode_communicator
        )
        self._bagua_global_communicator = self._bagua_backend.global_communicator
        self.bagua_communication_stream = self._bagua_backend.stream

        # autotune service
        from bagua.torch_api.communication import get_hyperparameters_service_client

        self._bagua_autotune_client = get_hyperparameters_service_client()

        self._bagua_init_algorithm()
        return self

    def _bagua_autotune_register_tensors(self):
        """
        Register tensors on autotune server, and return first bucketing suggestions
        """
        autotune_tensor_list = [
            TensorDeclaration(
                {
                    "name": tensor.bagua_tensor_name,
                    "num_elements": tensor.numel(),
                    "dtype": to_bagua_datatype(tensor.dtype),
                }
            )
            for tensor in self._bagua_tensors
        ]

        rsp = self._bagua_autotune_client.register_tensors(
            model_name=self.bagua_module_name, tensor_list=autotune_tensor_list
        )
        assert rsp.status_code == 200, "Unexpected rsp={}".format(rsp)

    def _bagua_autotune_get_buckets(self):
        rsp = self._bagua_autotune_client.ask_hyperparameters(
            model_name=self.bagua_module_name,
            rank=get_rank(),
            train_iter=self.bagua_train_step_counter,
        )
        assert rsp.status_code == 200, "Unexpected rsp={}".format(rsp)
        recommended_hyperparameters = rsp.json()["recommended_hyperparameters"]
        is_autotune_completed = rsp.json()["is_autotune_completed"]

        self._bagua_hyperparameters.update(recommended_hyperparameters)

        self._bagua_autotune_completed = is_autotune_completed
        recommended_buckets = map(
            lambda x: list(map(lambda y: self._bagua_tensor_map[y["name"]], x)),
            recommended_hyperparameters["buckets"],
        )
        return list(recommended_buckets)

    def _bagua_init_algorithm(self):
        self._bagua_cleanup_algorithm()
        self._bagua_broadcast_parameters()
        self._bagua_tensors = self.bagua_algorithm.init_tensors(self)
        self._bagua_tensor_map = dict(
            [(tensor.bagua_tensor_name, tensor) for tensor in self._bagua_tensors]
        )
        self._bagua_autotune_register_tensors()
        self._bagua_reset_algorithm_buckets()

    def _bagua_cleanup_algorithm(self):
        for hook in self._bagua_algorithm_hooks:
            hook.remove()
        self._bagua_algorithm_hooks.clear()
        self.bagua_buckets.clear()

    def _bagua_reset_algorithm_buckets(self):
        self._bagua_cleanup_algorithm()
        raw_buckets = self._bagua_autotune_get_buckets()
        self.bagua_buckets.extend(self.bagua_algorithm.tensors_to_buckets(raw_buckets))

        for name, param in self.named_parameters():

            def real_hook_factory(param_name, parameter):
                def real_hook(*unused):
                    self.bagua_algorithm.init_backward_hook(self)(param_name, parameter)

                    def real_post_backward_hook(*unused):
                        if self._speed_metrics_switch_on:
                            torch.cuda.current_stream().record_event(
                                self._speed_metrics_end_event
                            )

                        self.bagua_algorithm.init_post_backward_hook(self)()

                    if not self._is_post_backward_callback_queued:
                        torch.autograd.Variable._execution_engine.queue_callback(
                            real_post_backward_hook
                        )
                        self._is_post_backward_callback_queued = True

                return real_hook

            if param.requires_grad:
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                hook = grad_acc.register_hook(real_hook_factory(name, param))
                hook.grad_acc = grad_acc
                self._bagua_algorithm_hooks.append(hook)

        optimizer_hook = self.bagua_algorithm.init_post_optimizer_step_hook(self)

        from types import MethodType

        for optimizer in self.bagua_optimizers:
            if not hasattr(optimizer, "_bagua_original_step"):
                optimizer._bagua_original_step = optimizer.step

            def new_step_factory(optimizer):
                def new_step(self, *args, **kwargs):
                    result = self._bagua_original_step(*args, **kwargs)
                    optimizer_hook(self)
                    return result

                return MethodType(new_step, optimizer)

            optimizer.step = new_step_factory(optimizer)

        for bucket in self.bagua_buckets:
            self.bagua_algorithm.init_operations(
                self,
                bucket,
            )
        self._bagua_backend.register_ordered_buckets(
            [bucket.backend_bucket for bucket in self.bagua_buckets]
        )


_base = gorilla._get_base(BaguaModule)
_decorator_data = gorilla.get_decorator_data(_base)
for patch in _decorator_data.patches:
    gorilla.apply(patch)
