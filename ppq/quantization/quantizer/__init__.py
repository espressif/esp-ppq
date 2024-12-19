from .AscendQuantizer import AscendQuantizer
from .base import BaseQuantizer
from .DSPQuantizer import PPL_DSP_Quantizer, PPL_DSP_TI_Quantizer
from .EspdlQuantizer import EspdlInt16Quantizer, EspdlQuantizer, EspdlS3Quantizer, EspdlS3Int16Quantizer, EspdlHPreInt16Quantizer, EspdlS3HPreInt16Quantizer
from .FP8Quantizer import GraphCoreQuantizer, TensorRTQuantizer_FP8

# from .TRTQuantizer import TensorRTQuantizer
from .FPGAQuantizer import FPGAQuantizer
from .MetaxQuantizer import MetaxChannelwiseQuantizer, MetaxTensorwiseQuantizer
from .MNNQuantizer import MNNQuantizer
from .MyQuantizer import ExtQuantizer
from .NCNNQuantizer import NCNNQuantizer
from .NXPQuantizer import NXP_Quantizer
from .OpenvinoQuantizer import OpenvinoQuantizer
from .ORTQuantizer import OnnxruntimeQuantizer
from .PPLQuantizer import PPLCUDAQuantizer
from .RKNNQuantizer import RKNN_PerChannelQuantizer, RKNN_PerTensorQuantizer
from .TengineQuantizer import TengineQuantizer
from .TensorRTQuantizer import TensorRTQuantizer, TensorRTQuantizer_InputOnly
