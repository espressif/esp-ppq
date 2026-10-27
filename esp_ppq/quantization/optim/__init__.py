from .baking import ParameterBakingPass
from .base import QuantizationOptimizationPass, QuantizationOptimizationPipeline
from .calibration import (
    IsotoneCalibrationPass,
    PPLDSPTIReCalibrationPass,
    QuantConfigModifyPass,
    RuntimeCalibrationPass,
)
from .equalization import ActivationEqualizationPass, ChannelwiseSplitPass, LayerwiseEqualizationPass
from .espdl_attention import (
    EspdlAttentionFusionPass,
    EspdlAttentionLegalizePass,
    espdl_attention_platform_unsupported_reason,
    finalize_espdl_attention_for_export,
    legalize_espdl_attention_graph,
)
from .espdl_lut import EspdlLUTFusionPass
from .extension import ExtensionPass
from .inverse_sigmoid_fusion import (
    InverseSigmoidFusionPass,
    finalize_espdl_inverse_sigmoid_for_export,
    inverse_sigmoid_unsupported_reason,
)
from .legacy import AdaroundPass
from .morph import (
    ConvTransposeDecompositionPass,
    GRUSplitPass,
    HorizontalLayerSplitPass,
    MetaxGemmSplitPass,
    NCNNFormatGemmPass,
    NXPResizeModeChangePass,
)
from .parameters import ParameterQuantizePass, PassiveParameterQuantizePass
from .refine import (
    MishFusionPass,
    NxpInputRoundingRefinePass,
    NxpQuantizeFusionPass,
    QuantAlignmentPass,
    QuantizeFusionPass,
    QuantizeSimplifyPass,
    SwishFusionPass,
)
from .rnn_arity_normalization import RNNArityNormalizationPass
from .rotary_embedding_fusion import (
    EspdlRotaryEmbeddingLegalizePass,
    RotaryEmbeddingFusionPass,
    check_rotary_embedding,
    finalize_espdl_rotary_embedding_for_export,
    halve_rope_cache,
    legalize_espdl_rotary_embedding_graph,
)
from .ssd import SSDEqualizationPass
from .training import BiasCorrectionPass, LearnedStepSizePass, RoundTuningPass, TrainedQuantizationThresholdPass
