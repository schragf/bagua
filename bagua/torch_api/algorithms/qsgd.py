from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.algorithms import Algorithm
from bagua.torch_api.communication import get_backend
from typing import List
import torch
import bagua_core as B

from bagua.torch_api import get_world_size, get_rank, get_local_rank, get_local_size
from bagua.torch_api import allgather_inplace, alltoall_inplace, allgather, recv, send, reduce_inplace, ReduceOp, broadcast
import torch.distributed as dist
import torch.multiprocessing as mp
# compression:
import cupy
from cupy._binary import packing
from torch.utils.dlpack import to_dlpack, from_dlpack

import math


class QSGDAlgorithm(Algorithm):

    def __init__(self, quantization_bits: int = 8):
        self.quantization_bits = quantization_bits

    def init_operations(
        self,
        bagua_module: BaguaModule,
        bucket: BaguaBucket,
    ):

        bucket.clear_ops()

        def qsgd_centeralized_communication(*args):

            def compression_tensor(cat_tensor):
                scale = torch.norm(cat_tensor)
                # signs = torch.sign(cat_tensor).add(1.0).bool()
                signs = torch.sign(cat_tensor).add(1.0)
                normalized_tensor = cat_tensor / scale
                # for performance: just round to nearest level instead of random
                level = normalized_tensor * level_of_quantization
                quantized = torch.round(level).abs()
                quantized = torch.clamp(quantized, max=level_of_quantization)

                return scale, signs, quantized.byte()

            def compression_signs(signs, is_chunk):
                packed_size = (chunk_size + 7) // 8
                sign_padding = packed_size * 8 - chunk_size
                signs_bool = torch.empty(
                    0, dtype=torch.uint8, device='cuda:{}'.format(torch.cuda.current_device()))
                if (not is_chunk):
                    signs_padded = torch.empty(
                        0, device='cuda:{}'.format(torch.cuda.current_device()))
                    for i in range(n_workers):
                        temp = torch.cat((signs[:chunk_size], torch.zeros(
                            sign_padding, device='cuda:{}'.format(torch.cuda.current_device()))))
                        signs = signs[chunk_size:]
                        signs_padded = torch.cat((signs_padded, temp))
                    signs_bool = signs_padded.bool()
                else:
                    signs_bool = signs.bool()
                signs_dlpack = to_dlpack(signs_bool)
                signs_cupy = cupy.fromDlpack(signs_dlpack)
                signs = cupy.packbits(signs_cupy)
                signs = signs.toDlpack()
                signs = from_dlpack(signs).cuda()

                return signs, sign_padding

            def decompression(scale, compressed_signs, compressed_tensor):
                signs_cupy = cupy.fromDlpack(to_dlpack(compressed_signs))
                signs_cupy_unpacked = cupy.unpackbits(signs_cupy)
                signs_unpacked = signs_cupy_unpacked.toDlpack()
                signs_unpacked = from_dlpack(signs_unpacked).cuda()
                signs_decompressed_pad = signs_unpacked.float().mul(2.0).sub(1.0)
                signs = torch.empty(0, device='cuda:{}'.format(
                    torch.cuda.current_device()))
                for i in range(n_workers):
                    signs = torch.cat(
                        (signs, signs_decompressed_pad[:chunk_size]))
                    signs_decompressed_pad = signs_decompressed_pad[chunk_size + sign_padding:]
                signs = torch.reshape(signs, (n_workers, chunk_size))
                return compressed_tensor / level_of_quantization * scale * signs

            backend = get_backend("")
            rank = get_rank()
            n_nodes = get_world_size()
            n_workers = n_nodes
            # currently only works with 8 quantization_bits
            quantization_bits = 8
            level_of_quantization = (1 << quantization_bits) - 1

            cat_tensor = torch.empty(
                0, device='cuda:{}'.format(torch.cuda.current_device()))
            # Concatenate the tensors in the bucket
            for tensor in bucket.tensors:
                clone_tensor = tensor.clone().detach()
                cat_tensor = torch.cat(
                    (cat_tensor, clone_tensor.flatten()))

            # compress the cat_tensor
            scale, signs, compressed_tensor = compression_tensor(
                cat_tensor)
            del cat_tensor
            # allgather the sale of own compressed_tensor and receive all the other scales
            recv_scales = torch.empty(
                n_workers, device='cuda:{}'.format(torch.cuda.current_device()))
            allgather(scale, recv_scales)
            recv_scales = torch.reshape(recv_scales, (n_workers, 1))
            # prepare compressed_tensor for alltoall by adding padding if needed
            chunk_size = math.ceil(compressed_tensor.numel() / n_workers)
            # chunk_padding: number of elements to add to that compressed_tensor can be distributed evently to all workers
            chunk_padding = 0

            if (compressed_tensor.numel() % n_workers != 0):
                chunk_padding = (chunk_size * n_workers) - \
                    compressed_tensor.numel()
                compressed_tensor = torch.cat((compressed_tensor, torch.empty(
                    chunk_padding, dtype=torch.uint8, device='cuda:{}'.format(torch.cuda.current_device()))))
                signs = torch.cat((signs, torch.zeros(
                    chunk_padding, dtype=torch.uint8, device='cuda:{}'.format(torch.cuda.current_device()))))
            compressed_tensor = compressed_tensor.byte()
            alltoall_inplace(compressed_tensor)
            recv_tensor = torch.reshape(
                compressed_tensor, (n_workers, chunk_size)).float()
            del compressed_tensor
            # compress the signs and prepare them for alltoall by adding padding
            compressed_signs, sign_padding = compression_signs(
                signs, is_chunk=False)
            alltoall_inplace(compressed_signs)

            chunk_decompressed = decompression(
                recv_scales, compressed_signs, recv_tensor)
            del compressed_signs
            del recv_scales
            del recv_tensor

            chunk_avg = torch.mean(chunk_decompressed, 0)

            scale, signs, compressed_tensor = compression_tensor(chunk_avg)
            compressed_signs, sign_padding = compression_signs(
                signs, is_chunk=True)
            del chunk_avg

            # communicate averaged chunks
            recv_scales = torch.empty(
                (scale.numel() * n_workers), device='cuda:{}'.format(torch.cuda.current_device()))
            allgather(scale, recv_scales)
            recv_scales = torch.reshape(recv_scales, (n_workers, 1))
            del scale

            recv_signs = torch.empty((compressed_signs.numel(
            ) * n_workers), dtype=torch.uint8, device='cuda:{}'.format(torch.cuda.current_device()))
            allgather(compressed_signs, recv_signs)
            del compressed_signs

            recv_tensor = torch.empty((compressed_tensor.numel() * n_workers), dtype=torch.uint8,
                                      device='cuda:{}'.format(torch.cuda.current_device()))
            allgather(compressed_tensor, recv_tensor)
            del compressed_tensor

            # TESTING RESHAPING
            recv_scales = torch.reshape(recv_scales, (n_workers, 1))
            recv_tensor = torch.reshape(
                recv_tensor, (n_workers, chunk_size)).float()

            decompressed = decompression(
                recv_scales, recv_signs, recv_tensor)
            del recv_scales, recv_signs, recv_tensor
            decompressed = decompressed.flatten()
            padding = decompressed.numel() - chunk_padding
            update_tensor = decompressed[0:padding]
            for tensor in bucket.tensors:
                tensor_size = tensor.numel()
                tensor_shape = tensor.shape
                new_tensor = update_tensor[0:tensor_size]
                new_tensor = torch.reshape(new_tensor, tensor_shape)
                tensor.set_(new_tensor)
                update_tensor = update_tensor[tensor_size:]
            del update_tensor

            torch.cuda.empty_cache()

        bucket.append_python_op(qsgd_centeralized_communication)
