#!/usr/bin/env python3
"""Numerical validation of ConvTranspose decomposition."""

import torch
import torch.nn as nn
from pathlib import Path

from esp_ppq.api import load_onnx_graph
from esp_ppq.IR.morph import GraphDecomposer
from esp_ppq.executor import TorchExecutor


def create_convtranspose_model():
    """Create a simple model with ConvTranspose layer."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_transpose = nn.ConvTranspose2d(
                in_channels=3,
                out_channels=6,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=True
            )

        def forward(self, x):
            return self.conv_transpose(x)

    model = SimpleModel()
    model.eval()
    return model


def test_numerical_accuracy():
    print("Testing numerical accuracy of ConvTranspose decomposition...")

    # Create model and sample input
    model = create_convtranspose_model()
    sample_input = torch.randn(1, 3, 32, 32)

    # Get reference output from PyTorch
    with torch.no_grad():
        reference_output = model(sample_input)
    print(f"Reference output shape: {reference_output.shape}")

    # Export model to ONNX
    onnx_path = Path('test_convtranspose_numerical.onnx')
    torch.onnx.export(
        model,
        sample_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )

    # Load graph
    graph = load_onnx_graph(str(onnx_path))

    # Apply decomposition
    decomposer = GraphDecomposer(graph)
    decomposer.decompose_convtranspose()

    # Debug: print graph outputs
    print(f"Debug: graph outputs: {list(graph.outputs.keys())}")
    for var_name, var in graph.outputs.items():
        print(f"Debug: output var '{var_name}': source_op={var.source_op.name if var.source_op else None}")

    # Ensure output variable has a source operation
    assert 'output' in graph.outputs, "Output variable 'output' not found in graph outputs"
    output_var = graph.outputs['output']
    assert output_var.source_op is not None, f"Output variable 'output' has no source operation"

    # Debug: print Conv operation details
    conv_ops = [op for op in graph.operations.values() if op.type == 'Conv']
    for conv_op in conv_ops:
        print(f"Debug: Conv op '{conv_op.name}' inputs: {[var.name for var in conv_op.inputs]}")
        print(f"Debug: Conv op '{conv_op.name}' attributes: {conv_op.attributes}")
        for i, var in enumerate(conv_op.inputs):
            print(f"Debug:   input {i} '{var.name}': is_parameter={var.is_parameter}, value type={type(var.value)}, shape={var.value.shape if var.value is not None else None}")

    # Create executor and run decomposed graph
    executor = TorchExecutor(graph, device='cpu')
    decomposed_output = executor.forward(inputs=sample_input, output_names=['output'])
    print(f"Debug: decomposed_output type: {type(decomposed_output)}")
    if isinstance(decomposed_output, dict):
        print(f"Debug: decomposed_output keys: {list(decomposed_output.keys())}")
        print(f"Debug: decomposed_output values: {[(k, type(v)) for k, v in decomposed_output.items()]}")
    elif isinstance(decomposed_output, list):
        print(f"Debug: decomposed_output length: {len(decomposed_output)}")
        for i, item in enumerate(decomposed_output):
            print(f"Debug: decomposed_output[{i}] type: {type(item)}")
    else:
        print(f"Debug: decomposed_output: {decomposed_output}")

    # Convert output to tensor
    decomposed_output_tensor = decomposed_output[0]

    # Compare results
    print(f"Decomposed output shape: {decomposed_output_tensor.shape}")

    # Calculate absolute difference
    abs_diff = torch.abs(reference_output - decomposed_output_tensor)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()

    print(f"Numerical comparison:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")

    # Check if differences are within tolerance
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"✓ Numerical accuracy within tolerance ({tolerance})")
        success = True
    else:
        print(f"✗ Numerical accuracy exceeds tolerance ({tolerance})")
        success = False

    # Clean up
    onnx_path.unlink(missing_ok=True)

    return success


def test_various_configurations():
    """Test different ConvTranspose configurations."""
    test_cases = [
        {'in_channels': 3, 'out_channels': 6, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1},
        {'in_channels': 1, 'out_channels': 2, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'output_padding': 0},
        {'in_channels': 4, 'out_channels': 8, 'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 0},
        {'in_channels': 2, 'out_channels': 4, 'kernel_size': 3, 'stride': 2, 'padding': 0, 'output_padding': 1},
    ]

    all_success = True
    for i, config in enumerate(test_cases):
        print(f"\nTest case {i+1}: {config}")

        class TestModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.conv_transpose = nn.ConvTranspose2d(
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels'],
                    kernel_size=config['kernel_size'],
                    stride=config['stride'],
                    padding=config['padding'],
                    output_padding=config['output_padding'],
                    bias=True
                )

            def forward(self, x):
                return self.conv_transpose(x)

        model = TestModel(config)
        model.eval()

        # Create input with appropriate shape
        input_shape = (1, config['in_channels'], 16, 16)
        sample_input = torch.randn(input_shape)

        # Reference output
        with torch.no_grad():
            reference_output = model(sample_input)

        # Export and load graph
        onnx_path = Path(f'test_case_{i}.onnx')
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=11
        )

        graph = load_onnx_graph(str(onnx_path))

        # Apply decomposition
        decomposer = GraphDecomposer(graph)
        try:
            decomposer.decompose_convtranspose()
        except Exception as e:
            print(f"  ✗ Decomposition failed: {e}")
            all_success = False
            continue

        # Execute decomposed graph
        executor = TorchExecutor(graph, device='cpu')
        decomposed_output = executor.forward(inputs=sample_input, output_names=['output'])
        decomposed_output_tensor = decomposed_output[0]

        # Compare
        abs_diff = torch.abs(reference_output - decomposed_output_tensor)
        max_diff = torch.max(abs_diff).item()

        tolerance = 1e-5
        if max_diff < tolerance:
            print(f"  ✓ Passed (max diff: {max_diff:.2e})")
        else:
            print(f"  ✗ Failed (max diff: {max_diff:.2e})")
            all_success = False

        # Clean up
        onnx_path.unlink(missing_ok=True)

    return all_success


def test_conv1d_convtranspose():
    """Test Conv1D ConvTranspose decomposition."""
    test_cases = [
        {'in_channels': 3, 'out_channels': 6, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1},
        {'in_channels': 1, 'out_channels': 2, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'output_padding': 0},
        {'in_channels': 4, 'out_channels': 8, 'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 0},
        {'in_channels': 2, 'out_channels': 4, 'kernel_size': 3, 'stride': 2, 'padding': 0, 'output_padding': 1},
    ]

    all_success = True
    for i, config in enumerate(test_cases):
        print(f"\nTest Conv1D case {i+1}: {config}")

        class TestModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.conv_transpose = nn.ConvTranspose1d(
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels'],
                    kernel_size=config['kernel_size'],
                    stride=config['stride'],
                    padding=config['padding'],
                    output_padding=config['output_padding'],
                    bias=True
                )

            def forward(self, x):
                return self.conv_transpose(x)

        model = TestModel(config)
        model.eval()

        # Create input with appropriate shape
        input_shape = (1, config['in_channels'], 32)  # 1D input
        sample_input = torch.randn(input_shape)

        # Reference output
        with torch.no_grad():
            reference_output = model(sample_input)

        # Export and load graph
        onnx_path = Path(f'test_conv1d_case_{i}.onnx')
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=11
        )

        graph = load_onnx_graph(str(onnx_path))

        # Apply decomposition
        decomposer = GraphDecomposer(graph)
        try:
            decomposer.decompose_convtranspose()
        except Exception as e:
            print(f"  ✗ Decomposition failed: {e}")
            all_success = False
            continue

        # Execute decomposed graph
        executor = TorchExecutor(graph, device='cpu')
        decomposed_output = executor.forward(inputs=sample_input, output_names=['output'])
        decomposed_output_tensor = decomposed_output[0]

        # Compare
        abs_diff = torch.abs(reference_output - decomposed_output_tensor)
        max_diff = torch.max(abs_diff).item()

        tolerance = 1e-5
        if max_diff < tolerance:
            print(f"  ✓ Passed (max diff: {max_diff:.2e})")
        else:
            print(f"  ✗ Failed (max diff: {max_diff:.2e})")
            all_success = False

        # Clean up
        onnx_path.unlink(missing_ok=True)

    return all_success

if __name__ == '__main__':
    try:
        print("=" * 60)
        print("Numerical Validation of ConvTranspose Decomposition")
        print("=" * 60)

        success1 = test_numerical_accuracy()
        success2 = test_various_configurations()
        success3 = test_conv1d_convtranspose()

        if success1 and success2 and success3:
            print("\n" + "=" * 60)
            print("All numerical tests passed! ✓")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("Some tests failed! ✗")
            print("=" * 60)
            exit(1)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)