import itertools
import warnings
from typing import Any, Callable, Dict, Iterator, Literal, Union

import e3nn.o3 as o3
import numpy as np
import torch.cuda
import torch.nn

from .convolution import IrrepsConvolution, IrrepsScatterGatterFusedConvolution
from .linear import IrrepsLinear
from .self_connection import SelfConnectionIntro, SelfConnectionLinearIntro

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    _CUE_AVAILABLE = True

    # Obatained from MACE
    class O3_e3nn(cue.O3):
        def __mul__(  # type: ignore
            rep1: 'O3_e3nn', rep2: 'O3_e3nn'
        ) -> Iterator['O3_e3nn']:
            return [  # type: ignore
                O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)
            ]

        @classmethod
        def clebsch_gordan(  # type: ignore
            cls, rep1: 'O3_e3nn', rep2: 'O3_e3nn', rep3: 'O3_e3nn'
        ) -> np.ndarray:
            rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

            if rep1.p * rep2.p == rep3.p:
                return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(
                    rep3.dim
                )
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

        def __lt__(  # type: ignore
            rep1: 'O3_e3nn', rep2: 'O3_e3nn'
        ) -> bool:
            rep2 = rep1._from(rep2)  # type: ignore
            return (rep1.l, rep1.p) < (rep2.l, rep2.p)

        @classmethod
        def iterator(cls) -> Iterator['O3_e3nn']:
            for l in itertools.count(0):
                yield O3_e3nn(l=l, p=1 * (-1) ** l)
                yield O3_e3nn(l=l, p=-1 * (-1) ** l)

    class CueqFCTPLinear(torch.nn.Module):
        """Collapsed FCTP for a one-hot scalar second operand."""

        def __init__(
            self,
            irreps_in1: cue.Irreps,
            irreps_in2: cue.Irreps,
            irreps_out: cue.Irreps,
            *,
            shared_weights: bool = True,
            internal_weights: Union[bool, None] = None,
            method: str = 'indexed_linear',
            **kwargs,
        ) -> None:
            super().__init__()

            if not shared_weights or internal_weights is False:
                raise ValueError('CueqFCTPLinear requires shared internal weights')
            if method != 'indexed_linear':
                raise ValueError('CueqFCTPLinear requires method="indexed_linear"')
            if len(irreps_in2) != 1:
                raise ValueError('The second operand must contain one scalar irrep')

            num_elements, operand_irrep = irreps_in2[0]
            if operand_irrep.l != 0 or getattr(operand_irrep, 'p', 1) != 1:
                raise ValueError('The second operand must be an even scalar irrep')

            in_irreps = [ir for _, ir in irreps_in1]
            out_irreps = [ir for _, ir in irreps_out]
            if any(ir in in_irreps[:i] for i, ir in enumerate(in_irreps)):
                raise ValueError('Repeated input irreps are not supported')
            if any(ir in out_irreps[:i] for i, ir in enumerate(out_irreps)):
                raise ValueError('Repeated output irreps are not supported')

            self.num_elements = int(num_elements)
            self.irreps_out_dim = irreps_out.dim
            output_mask = [ir in in_irreps for _, ir in irreps_out]
            self.output_segments = [
                (keep, int(mul * ir.dim))
                for keep, (mul, ir) in zip(output_mask, irreps_out)
            ]
            linear_irreps_out = irreps_out.filter(mask=output_mask)
            self.path_shapes = [
                (int(mul_in), self.num_elements, int(mul_out))
                for mul_in, ir_in in irreps_in1
                for mul_out, ir_out in irreps_out
                if ir_in == ir_out
            ]
            self.weight_numel = sum(
                mul_in * num_elem * mul_out
                for mul_in, num_elem, mul_out in self.path_shapes
            )
            linear_weight_numel = sum(
                mul_in * mul_out
                for mul_in, _, mul_out in self.path_shapes
            )

            device = kwargs.get('device')
            dtype = kwargs.get('dtype')
            self.weight = torch.nn.Parameter(
                torch.randn(self.weight_numel, device=device, dtype=dtype)
            )

            self.linear = None
            if linear_weight_numel > 0:
                self.linear = cuet.Linear(
                    irreps_in1,
                    linear_irreps_out,
                    shared_weights=True,
                    internal_weights=False,
                    weight_classes=self.num_elements,
                    method=method,
                    **kwargs,
                )
                if self.linear.weight_numel != linear_weight_numel:
                    raise ValueError('Unexpected cuEquivariance linear path layout')

        def _pack_weights(self) -> torch.Tensor:
            blocks = []
            offset = 0
            for mul_in, num_elements, mul_out in self.path_shapes:
                size = mul_in * num_elements * mul_out
                block = self.weight[offset : offset + size].reshape(
                    mul_in, num_elements, mul_out
                )
                blocks.append(block.permute(1, 0, 2).reshape(num_elements, -1))
                offset += size

            if not blocks:
                return self.weight.new_empty((self.num_elements, 0))
            return torch.cat(blocks, dim=1) / self.num_elements**0.5

        def forward(self, x: torch.Tensor, operand: torch.Tensor) -> torch.Tensor:
            if operand.ndim != 2 or operand.shape[1] != self.num_elements:
                raise ValueError('Operand shape must be [num_atoms, num_elements]')
            if x.shape[0] != operand.shape[0]:
                raise ValueError('Input and operand atom counts must match')
            if self.linear is None or x.shape[0] == 0:
                zero = x.sum(dim=1, keepdim=True) * 0.0 + self.weight.sum() * 0.0
                return zero.expand(x.shape[0], self.irreps_out_dim)

            indices = torch.argmax(operand, dim=1)
            order = torch.argsort(indices, stable=True)
            inverse_order = torch.argsort(order)
            output = self.linear(
                x.index_select(0, order),
                self._pack_weights(),
                indices.index_select(0, order),
            )
            output = output.index_select(0, inverse_order)

            output_blocks = []
            offset = 0
            zero = x.sum(dim=1, keepdim=True) * 0.0
            for keep, dim in self.output_segments:
                if keep:
                    output_blocks.append(output[:, offset : offset + dim])
                    offset += dim
                else:
                    output_blocks.append(zero.expand(x.shape[0], dim))
            return torch.cat(output_blocks, dim=1)

    class cueq_fused_scatter_channelwise_conv(torch.nn.Module):
        def __init__(
            self,
            irreps_in1: cue.Irreps,  # node feats
            irreps_in2: cue.Irreps,  # spherical harmonics
            irreps_out: cue.Irreps,
            tp_method: str,
            in_out_irreps_layout: cue.IrrepsLayout = cue.IrrepsLayout.mul_ir,
            **kwargs_dct,
        ) -> None:
            super().__init__()

            irreps_x = irreps_in1
            irreps_filter = irreps_in2

            assert not kwargs_dct.pop('shared_weights', False)
            assert not kwargs_dct.pop('internal_weights', False)

            self.t_in = cuet.TransposeIrrepsLayout(
                irreps_x,
                source=in_out_irreps_layout,
                target=cue.IrrepsLayout.ir_mul,
            )

            self.t_out = cuet.TransposeIrrepsLayout(
                irreps_out,
                source=cue.IrrepsLayout.ir_mul,
                target=in_out_irreps_layout,
            )

            x_muls = np.array(irreps_x.muls)
            if np.all(x_muls == x_muls[0]):
                # all channel same, simple fusion
                poly = (
                    cue.descriptors.channelwise_tensor_product(
                        irreps_x,
                        irreps_filter,
                        irreps_out,
                    )
                    .flatten_coefficient_modes()
                    .squeeze_modes()
                    .polynomial
                )
            else:
                assert in_out_irreps_layout == cue.IrrepsLayout.mul_ir
                # https://github.com/NVIDIA/cuEquivariance/issues/112
                n = np.gcd.reduce(x_muls)
                poly = cue.descriptors.channelwise_tensor_product(
                    irreps_x, irreps_filter, irreps_out
                ).polynomial
                poly = poly.flatten_modes('ijk').squeeze_modes('v')
                poly = poly.apply_fn(lambda op, d: (op, d.split_mode('u', n)))
                assert poly.all_same_segment_shape()

            self.f = cuet.SegmentedPolynomial(poly, method=tp_method)

        def forward(
            self,
            x: torch.Tensor,
            filter: torch.Tensor,
            w: torch.Tensor,
            edge_src: torch.Tensor,
            edge_dst: torch.Tensor,
        ) -> torch.Tensor:
            xt = self.t_in(x)
            o = self.f(
                [w, xt, filter],
                {1: edge_src.to(torch.int64)},
                {0: xt},
                {0: edge_dst.to(torch.int64)},
            )[0]
            return self.t_out(o)


except ImportError:
    _CUE_AVAILABLE = False


def is_cue_available() -> bool:
    return _CUE_AVAILABLE and torch.cuda.is_available()


def cue_needed(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if is_cue_available():
            return func(*args, **kwargs)
        else:
            raise ImportError('cue is not available')

    return wrapper


def _check_may_not_compatible(orig_kwargs, defaults) -> None:
    for k, v in defaults.items():
        v_given = orig_kwargs.pop(k, v)
        if v_given != v:
            warnings.warn(f'{k}: {v} is ignored to use cuEquivariance')


def is_cue_cuda_available_model(config: Dict[str, Any]) -> bool:
    if config.get('use_bias_in_linear', False):
        warnings.warn('Bias in linear can not be used with cueq, fallback to e3nn')
        return False
    else:
        return True


@cue_needed
def as_cue_irreps(irreps: o3.Irreps, group: Literal['SO3', 'O3']):
    """Convert e3nn irreps to given group's cue irreps"""
    if group == 'SO3':
        assert all(irrep.ir.p == 1 for irrep in irreps)
        return cue.Irreps('SO3', str(irreps).replace('e', ''))  # type: ignore
    elif group == 'O3':
        return cue.Irreps(O3_e3nn, str(irreps))  # type: ignore
    else:
        raise ValueError(f'Unknown group: {group}')


@cue_needed
def patch_linear(
    module: Union[IrrepsLinear, SelfConnectionLinearIntro],
    group: Literal['SO3', 'O3'],
    **cue_kwargs,
) -> Union[IrrepsLinear, SelfConnectionLinearIntro]:
    assert not module.layer_instantiated

    module.irreps_in = as_cue_irreps(module.irreps_in, group)  # type: ignore
    module.irreps_out = as_cue_irreps(module.irreps_out, group)  # type: ignore

    orig_kwargs = module.linear_kwargs

    may_not_compatible_default = dict(
        f_in=None,
        f_out=None,
        instructions=None,
        biases=False,
        path_normalization='element',
        _optimize_einsums=None,
    )
    # pop may_not_compatible_defaults
    _check_may_not_compatible(orig_kwargs, may_not_compatible_default)

    module.linear_cls = cuet.Linear  # type: ignore
    orig_kwargs.update(**cue_kwargs)
    return module


@cue_needed
def patch_convolution(
    module: IrrepsConvolution,
    group: Literal['SO3', 'O3'],
    *,
    use_scatter_fusion: bool = False,
    tp_method: str = 'uniform_id',
    **cue_kwargs,
):
    assert not module.layer_instantiated

    if use_scatter_fusion:
        module = IrrepsScatterGatterFusedConvolution.from_irreps_convolution(module)

    # conv_kwargs patched in place
    conv_kwargs: dict = module.convolution_kwargs  # type: ignore

    for ir_k in ('irreps_in1', 'irreps_in2', 'irreps_out'):
        conv_kwargs[ir_k] = as_cue_irreps(conv_kwargs[ir_k], group)

    # Validation logic, old instance w/o instruction sort is not compatible
    inst_orig = conv_kwargs.pop('instructions')
    inst_sorted = sorted(inst_orig, key=lambda x: x[2])
    assert all([a == b for a, b in zip(inst_orig, inst_sorted)])

    if not use_scatter_fusion:
        conv_kwargs['filter_irreps_out'] = conv_kwargs.pop('irreps_out')
        may_not_compatible_default = dict(
            in1_var=None,
            in2_var=None,
            out_var=None,
            irrep_normalization=False,
            path_normalization='element',
            compile_left_right=True,
            compile_right=False,
            _specialized_code=None,
            _optimize_einsums=None,
        )
        # pop may_not_compatible_defaults
        _check_may_not_compatible(conv_kwargs, may_not_compatible_default)

        module.convolution_cls = cuet.ChannelWiseTensorProduct  # type: ignore
        conv_kwargs.update(**cue_kwargs)
    else:
        conv_kwargs.update(dict(tp_method=tp_method))
        module.convolution_cls = cueq_fused_scatter_channelwise_conv

    return module


@cue_needed
def patch_fully_connected(
    module: SelfConnectionIntro,
    group: Literal['SO3', 'O3'],
    **cue_kwargs,
) -> SelfConnectionIntro:
    assert not module.layer_instantiated

    module.irreps_in1 = as_cue_irreps(module.irreps_in1, group)  # type: ignore
    module.irreps_in2 = as_cue_irreps(module.irreps_in2, group)  # type: ignore
    module.irreps_out = as_cue_irreps(module.irreps_out, group)  # type: ignore

    may_not_compatible_default = dict(
        irrep_normalization=None,
        path_normalization=None,
    )
    # pop may_not_compatible_defaults
    _check_may_not_compatible(
        module.fc_tensor_product_kwargs, may_not_compatible_default
    )

    module.fc_tensor_product_cls = CueqFCTPLinear  # type: ignore
    module.fc_tensor_product_kwargs.update(
        method='indexed_linear',
        **cue_kwargs,
    )
    return module
