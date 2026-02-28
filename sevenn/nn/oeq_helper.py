import torch

from sevenn.nn.convolution import IrrepsConvolution

try:
    from openequivariance import (
        TensorProductConv,
        TPProblem,
        torch_to_oeq_dtype,
    )

    _OEQ_AVAILABLE = True
except ImportError:
    _OEQ_AVAILABLE = False


def is_oeq_available() -> bool:
    return _OEQ_AVAILABLE and torch.cuda.is_available()


class OEQConvolution(torch.nn.Module):
    """
    Wrapper around openequivariance.TensorProductConv to match
    IrrepsScatterGatterFusedConvolution.convolution_cls interface.
    Input kwargs are directly from e3nn TensorProduct.
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        shared_weights: bool = False,
        internal_weights: bool = False,
    ):
        super().__init__()
        self.dtype = torch.get_default_dtype()
        tpp = TPProblem(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            irrep_dtype=torch_to_oeq_dtype(self.dtype),
            weight_dtype=torch_to_oeq_dtype(self.dtype),
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )
        self.tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    def forward(self, x, edge_filter, weight, edge_src, edge_dst):
        # OEQ rows=dst, cols=src (swapped vs sevennet's arg order)
        # sevennet casts to int32 for flashTP; OEQ needs int64
        return self.tp_conv(
            x.to(self.dtype),
            edge_filter.to(self.dtype),
            weight.to(self.dtype),
            edge_dst.to(torch.int64),  # rows = dst
            edge_src.to(torch.int64),  # cols = src
        )


# TODO: temporarily use convolution block from flash impl
def patch_convolution(irreps_convolution: IrrepsConvolution):
    from sevenn.nn.convolution import IrrepsScatterGatterFusedConvolution

    assert not irreps_convolution.layer_instantiated  # TODO: ??
    ret = IrrepsScatterGatterFusedConvolution.from_irreps_convolution(
        irreps_convolution
    )
    ret.convolution_cls = OEQConvolution
    return ret
