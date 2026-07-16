import os
import warnings

if os.path.dirname(os.path.realpath(__file__)) == os.path.join(os.path.realpath(os.getcwd()), "esp_ppq"):
    message = "You are importing esp_ppq within its own root folder ({}). "
    warnings.warn(message.format(os.getcwd()))

# This file defines export functions & class of esp_ppq.
from esp_ppq.api.setting import (
    ActivationQuantizationSetting,
    DispatchingTable,
    EqualizationSetting,
    GraphFormatSetting,
    LSQSetting,
    LUTOptimizationSetting,
    ParameterQuantizationSetting,
    QuantizationFusionSetting,
    QuantizationSetting,
    QuantizationSettingFactory,
    TemplateSetting,
    TQTSetting,
)
from esp_ppq.core import *
from esp_ppq.executor import BaseGraphExecutor, TorchExecutor, TorchQuantizeDelegator

# ---------------------------------------------------------------------------
# ESP-DL bit-exact INT16 LUT activation support.
#
# Registering the ``LUT`` operation handler makes esp-ppq's simulation match the
# on-device step-interpolated / nearest-neighbour LUT computed by esp-dl, which
# solves the "activation exported as LUT cannot be aligned with hardware" issue.
# Registration is deferred to here so that both the executor and parser packages
# are fully initialized (avoids import cycles).
# ---------------------------------------------------------------------------
from esp_ppq.executor.op.torch.espdl_lut import (  # noqa: E402
    GlobalMode as LUTGlobalMode,
)
from esp_ppq.executor.op.torch.espdl_lut import (
    HardwareEmulator as LUTHardwareEmulator,
)
from esp_ppq.executor.op.torch.espdl_lut import (
    SimulationMode as LUTSimulationMode,
)
from esp_ppq.executor.op.torch.espdl_lut import (
    register_lut_op_handler,
)
from esp_ppq.executor.op.torch.espdl_lut import (
    set_simulation_mode as set_lut_simulation_mode,
)
from esp_ppq.IR import (
    BaseGraph,
    GraphBuilder,
    GraphCommand,
    GraphExporter,
    GraphFormatter,
    Operation,
    QuantableGraph,
    SearchableGraph,
    TrainableGraph,
    Variable,
)
from esp_ppq.IR.deploy import RunnableGraph
from esp_ppq.IR.quantize import QuantableOperation, QuantableVariable
from esp_ppq.log import NaiveLogger
from esp_ppq.quantization.analyse import (
    graphwise_error_analyse,
    layerwise_error_analyse,
    parameter_analyse,
    statistical_analyse,
    variable_analyse,
)
from esp_ppq.quantization.measure import (
    torch_cosine_similarity,
    torch_cosine_similarity_as_loss,
    torch_KL_divergence,
    torch_mean_square_error,
    torch_snr_error,
)
from esp_ppq.quantization.optim import (
    BiasCorrectionPass,
    EspdlLUTFusionPass,
    GRUSplitPass,
    HorizontalLayerSplitPass,
    LayerwiseEqualizationPass,
    MetaxGemmSplitPass,
    MishFusionPass,
    NxpInputRoundingRefinePass,
    NxpQuantizeFusionPass,
    NXPResizeModeChangePass,
    ParameterBakingPass,
    ParameterQuantizePass,
    PassiveParameterQuantizePass,
    QuantizationOptimizationPass,
    QuantizationOptimizationPipeline,
    QuantizeFusionPass,
    QuantizeSimplifyPass,
    RuntimeCalibrationPass,
    SwishFusionPass,
)
from esp_ppq.quantization.qfunction import (
    BaseQuantFunction,
    PPQDyamicLinearQuantFunction,
    PPQFloatingQuantFunction,
    PPQLinearQuant_toInt,
    PPQLinearQuantFunction,
    PPQuantFunction,
    PPQuantFunction_toInt,
)
from esp_ppq.quantization.quantizer import (
    BaseQuantizer,
    NXP_Quantizer,
    PPL_DSP_Quantizer,
    PPLCUDAQuantizer,
    TensorRTQuantizer,
)
from esp_ppq.scheduler import AggresiveDispatcher, ConservativeDispatcher, GraphDispatcher, PPLNNDispatcher
from esp_ppq.scheduler.perseus import Perseus
from esp_ppq.utils.round import ppq_numerical_round, ppq_round_to_power_of_2, ppq_tensor_round

register_lut_op_handler()
