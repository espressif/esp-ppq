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
    ParameterQuantizationSetting,
    QuantizationFusionSetting,
    QuantizationSetting,
    QuantizationSettingFactory,
    TemplateSetting,
)
from esp_ppq.core import *
from esp_ppq.executor import BaseGraphExecutor, TorchExecutor, TorchQuantizeDelegator
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
