from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from typing import List
import torch

from bagua.torch_api import get_world_size
from bagua.torch_api import allgather_inplace, alltoall_inplace
import torch.distributed as dist
import torch.multiprocessing as mp
# compression:
import cupy
from cupy._binary import packing
from torch.utils.dlpack import to_dlpack, from_dlpack

import math


class SignSGDAlgorithm(Algorithm):

    def init_operations(
        self,
        bagua_module: BaguaModule,
        bucket: BaguaBucket,
    ):
        """
        concatenate all the tensors of a bucket (cat_tensor),
        compress the tensor (1bit/sign compression),
        communicate via bagua backend all_to_all,
        decompress the received tensor,
        average the tensor,
        compress the tensor (1bit/sign compression),
        communicated average via bagua backend all_gather,
        decompress the received tensor,
        remove padding,
        update the bucket tensors
        """
        bucket.clear_ops()

        def onebit_centeralized_communication(*args):

            def onebit_compression(tensor):
                """
                - Compress tensor by taking the sign of each element and pack each signbit into a 8bit array
                Input:
                    - tensor: torch.tensor
                Output:
                    - sign_padding: number of padding bits needed at the last element of 8bit-tensor to fit all signbits
                    - compressed_tensor: torch.tensor (dtype=torch.uint8)
                """
                tensor_size = tensor.numel()
                signs = torch.sign(tensor)
                signs_binary = signs.add(1.0).bool()
                signs_binary_dlpack = to_dlpack(signs_binary)
                signs_binary_cupy = cupy.fromDlpack(signs_binary_dlpack)
                packed_size = (tensor_size + 7) // 8
                sign_padding = packed_size * 8 - tensor_size
                compressed_signs = cupy.packbits(signs_binary_cupy)
                compressed_signs = compressed_signs.toDlpack()
                compressed_tensor = from_dlpack(compressed_signs).cuda()

                return sign_padding, compressed_tensor

            def onebit_decompression(tensor):
                """
                - decompress tensor by unpacking the uint8 array and transforming the values: (0 -> -1, 1 -> +1)
                Input:
                    - tensor: torch.tensor (dtype=torch.uint8)
                Output:
                    - decompressed: torch.tensor (dtype=torch.float32)
                """
                tensor_cupy = cupy.fromDlpack(to_dlpack(tensor))
                tensor_cupy_unpacked = cupy.unpackbits(tensor_cupy)
                tensor_unpacked = tensor_cupy_unpacked.toDlpack()
                tensor_unpacked = from_dlpack(tensor_unpacked).cuda()
                decompressed = tensor_unpacked.float().mul(2.0).sub(1.0)
                return decompressed

            # get number of workers
            n_workers = get_world_size()

            cat_tensor = torch.empty(0, device='cuda')
            # Concatenate the tensors in the bucket
            for idx, tensor in enumerate(bucket.tensor):
                clone_tensor = tensor.clone().detach()
                cat_tensor = torch.cat((cat_tensor, clone_tensor.flatten()))

            cat_tensor_size = cat_tensor.numel()

            sign_padding, compressed_tensor = onebit_compression(cat_tensor)

            compressed_tensor_size = compressed_tensor.numel()

            # COMMUNICATION
            # ASSUMPTION: all_to_all will send the last chunk to the worker with the highest rank

            chunk_size = math.ceil(compressed_tensor_size / n_workers)
            # add padding to compressed_tensor, so that it can be used by bagua alltoall
            if (compressed_tensor_size % n_workers != 0):
                chunk_padding = (chunk_size * n_workers) - \
                    compressed_tensor_size
                sign_padding += chunk_padding * 8
                compressed_tensor = torch.cat((compressed_tensor, torch.empty(
                    chunk_padding, dtype=torch.uint8, device='cuda')))

            send_tensor = compressed_tensor
            alltoall_inplace(send_tensor)
            recv_tensor = send_tensor

            decompressed = onebit_decompression(recv_tensor)

            chunk_decompressed = torch.reshape(
                decompressed, (n_workers, chunk_size*8))

            # AVERAGE
            chunk_avg = torch.mean(chunk_decompressed, 0)

            _, compressed_chunk_avg = onebit_compression(chunk_avg)

            send_tensor = compressed_chunk_avg
            allgather_inplace(send_tensor)
            recv_tensor = send_tensor

            decompressed = onebit_decompression(recv_tensor)

            # UPDATE BUCKET
            # remove padding
            padding = decompressed.numel() - sign_padding
            update_tensor = decompressed[0:padding]

            for tensor in bucket.tensor:
                tensor_size = tensor.numel()
                tensor_shape = tensor.shape
                new_tensor = update_tensor[0:tensor_size]
                tensor._bagua_backend_tensor.torch_tensor.set_(
                    torch.reshape(new_tensor, tensor_shape))
                update_tensor = update_tensor[tensor_size:]

        bucket.append_python_op(onebit_centeralized_communication)
