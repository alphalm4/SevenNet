"""Tests for SevenNetModel torchsim integration.

This test file is structured to work in the torchsim repository's test directory.
It uses the factory function pattern from torchsim's test infrastructure.
"""

import pytest
import torch

try:
    from torch_sim.models.interface import validate_model_outputs
    from torch_sim.testing import (
        SIMSTATE_BULK_GENERATORS,
        assert_model_calculator_consistency,
    )

    TORCH_SIM_AVAILABLE = True
except ImportError:
    TORCH_SIM_AVAILABLE = False  # pyright: ignore[reportConstantRedefinition]
    SIMSTATE_BULK_GENERATORS = {}  # pyright: ignore[reportConstantRedefinition]

    def validate_model_outputs(*args, **kwargs):
        return None

    def assert_model_calculator_consistency(*args, **kwargs):
        return None

if not TORCH_SIM_AVAILABLE:
    pytest.skip(
        'torch_sim not installed. Install torch-sim-atomistic separately if needed.',
        allow_module_level=True,
    )

import sevenn.util
from sevenn.calculator import SevenNetCalculator
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.torchsim import SevenNetD3Model, SevenNetD4Model, SevenNetModel

# Test configuration
model_name = 'sevennet-mf-ompa'
modal_name = 'mpa'
DTYPE = torch.float32
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@pytest.fixture
def pretrained_sevenn_model() -> AtomGraphSequential:
    """Load a pretrained SevenNet model for testing."""
    cp = sevenn.util.load_checkpoint(model_name)
    model_loaded = cp.build_model()
    model_loaded.set_is_batch_data(True)  # pyright: ignore[reportCallIssue]
    return model_loaded.to(DEVICE)


@pytest.fixture
def sevenn_model(pretrained_sevenn_model: AtomGraphSequential) -> SevenNetModel:
    """Create a SevenNetModel wrapper for the pretrained model."""
    return SevenNetModel(
        model=pretrained_sevenn_model, modal=modal_name, device=DEVICE, dtype=DTYPE
    )


@pytest.fixture
def sevenn_calculator() -> SevenNetCalculator:
    """Create a SevenNetCalculator for consistency testing."""
    return SevenNetCalculator(model_name, modal=modal_name, device=str(DEVICE))


def test_sevenn_model_output_validation(sevenn_model: SevenNetModel) -> None:
    """Test that a model implementation follows the ModelInterface contract."""
    validate_model_outputs(sevenn_model, DEVICE, DTYPE)


def test_sevenn_model_dtype_validation(
    pretrained_sevenn_model: AtomGraphSequential
) -> None:
    """Test that SevenNetModel raises an error if dtype is not float32."""
    with pytest.raises(
        ValueError,
        match='SevenNet currently only supports torch.float32, but received '
        + 'different dtype: torch.float64',
    ):
        _ = SevenNetModel(
            model=pretrained_sevenn_model,
            modal=modal_name,
            device=DEVICE,
            dtype=torch.float64,
        )


@pytest.mark.parametrize('sim_state_name', SIMSTATE_BULK_GENERATORS)
def test_sevenn_model_consistency(
    sim_state_name: str,
    sevenn_model: SevenNetModel,
    sevenn_calculator: SevenNetCalculator,
) -> None:
    """Test consistency between SevenNetModel and SevenNetCalculator.

    NOTE: sevenn is broken for the benzene simstate is ase comparison."""
    sim_state = SIMSTATE_BULK_GENERATORS[sim_state_name](DEVICE, DTYPE)
    assert_model_calculator_consistency(sevenn_model, sevenn_calculator, sim_state)


@pytest.mark.parametrize('d3_mode', ['batch', 'serial'])
def test_sevennet_d3_cuda_dispersion(d3_mode):
    cuda_d3 = pytest.importorskip('cuda_d3')
    torch_api = pytest.importorskip('cuda_d3.torch_api')
    if not hasattr(torch_api.CudaD3Torch, 'compute_batch_packed'):
        pytest.skip('CUDA-Dispersion 0.2.0 or later is required')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip('CUDA-Dispersion requires a GPU with compute capability 8.0+')

    import torch_sim as ts
    from ase.build import bulk
    from torch_sim.elastic import voigt_6_to_full_3x3_stress

    structures = [
        bulk('Si', 'diamond', a=5.43, cubic=True),
        bulk('Cu', 'fcc', a=3.58, cubic=True).repeat((2, 2, 1)),
    ]
    structures[1].rattle(stdev=0.03, seed=317)
    model = SevenNetD3Model(
        '7net-0',
        device='cuda',
        d3_backend='cuda-dispersion',
        d3_mode=d3_mode,
    )
    state = ts.initialize_state(structures, device=model.device, dtype=model.dtype)
    bare = model.sevennet(state)
    actual = model(state)

    d3 = cuda_d3.CudaD3()
    reference = [d3.compute(atoms) for atoms in structures]
    correction = {
        'energy': torch.tensor([r['energy'] for r in reference]),
        'forces': torch.cat([torch.from_numpy(r['forces']) for r in reference]),
        'stress': voigt_6_to_full_3x3_stress(
            torch.stack([torch.from_numpy(r['stress']) for r in reference])
        ),
    }
    for name, result in actual.items():
        expected = bare[name] + correction[name].to(bare[name])
        torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('periodic', [True, False])
def test_sevennet_d4_cuda_dispersion(periodic):
    cuda_d4 = pytest.importorskip('cuda_d4')
    torch_api = pytest.importorskip('cuda_d4.torch_api')
    if not hasattr(torch_api.CudaD4Torch, 'compute_batch_packed'):
        pytest.skip('CUDA-Dispersion must provide the packed D4 CUDA API')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip('CUDA-Dispersion requires a GPU with compute capability 8.0+')

    import torch_sim as ts
    from ase.build import bulk, molecule
    from torch_sim.elastic import voigt_6_to_full_3x3_stress

    structures = (
        [bulk('Si', cubic=True), bulk('NaCl', 'rocksalt', a=5.64).repeat((2, 2, 3))]
        if periodic else [molecule('H2O'), molecule('NH3')]
    )
    structures[1].rattle(stdev=0.03, seed=317)
    options = {'disp2_cutoff': 30.0, 'disp3_cutoff': 20.0}
    model = SevenNetD4Model('7net-0', d4_options=options)
    state = ts.initialize_state(structures, device=model.device, dtype=model.dtype)
    state.system_extras['total_charge'] = torch.tensor(
        [0.0, 1.0], device=model.device
    )
    with pytest.raises(TypeError, match='neutral systems'):
        model(state, charges=[-1.0, 0.0])
    bare = SevenNetModel.forward(model, state)
    reference = cuda_d4.CudaD4(**options).compute_batch(structures)
    correction = {
        'energy': torch.tensor([r['energy'] for r in reference]),
        'forces': torch.cat([torch.from_numpy(r['forces']) for r in reference]),
        'stress': torch.zeros_like(bare['stress']),
    }
    if periodic:
        correction['stress'] = voigt_6_to_full_3x3_stress(
            torch.stack([torch.from_numpy(r['stress']) for r in reference])
        )
    for name, result in model(state).items():
        expected = bare[name] + correction[name].to(bare[name])
        torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
