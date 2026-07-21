import inspect
import json

from esp_ppq.api.setting import LUTOptimizationSetting, QuantizationSettingFactory
from esp_ppq.autoquant.interface import espdl_auto_quantize_onnx
from esp_ppq.autoquant.quantize import build_quant_setting
from esp_ppq.autoquant.save import save_experiment
from esp_ppq.autoquant.setting import AutoQuantSearchSetting


def test_lut_step_is_adjusted_to_nearest_power_of_two(monkeypatch):
    warnings = []
    monkeypatch.setattr("esp_ppq.api.setting.ppq_warning", warnings.append)
    setting = LUTOptimizationSetting()

    setting.step = 48

    assert setting.step == 64
    assert warnings == [
        "LUT optimization step 48 is not a power of two; it has been adjusted to the nearest power of two: 64."
    ]


def test_lut_step_loaded_from_json_is_adjusted(monkeypatch):
    warnings = []
    monkeypatch.setattr("esp_ppq.api.setting.ppq_warning", warnings.append)
    setting = QuantizationSettingFactory.espdl_setting()
    setting_dict = json.loads(setting.to_json())
    setting_dict["lut_optimization_setting"]["step"] = 24

    loaded = QuantizationSettingFactory.from_json(setting_dict)

    assert loaded.lut_optimization_setting.step == 32
    assert any("step 24 is not a power of two" in warning for warning in warnings)


def test_autoquant_keeps_single_compatible_setting_parameter():
    parameters = inspect.signature(espdl_auto_quantize_onnx).parameters

    assert "setting" in parameters
    assert "auto_setting" not in parameters
    assert parameters["setting"].annotation is AutoQuantSearchSetting


def test_autoquant_search_setting_accepts_base_quantization_setting():
    setting = AutoQuantSearchSetting()
    base_setting = QuantizationSettingFactory.espdl_setting()

    setting.quant_setting = base_setting

    assert setting.quant_setting is base_setting


def test_autoquant_candidate_copies_base_quantization_setting():
    base_setting = QuantizationSettingFactory.espdl_setting()
    base_setting.lsq_optimization = True
    base_setting.quantize_activation_setting.calib_algorithm = "percentile"

    strategy = {
        "weight_equalization": {"value": False},
        "horizontal_layer_split": {"value": False},
        "bias_correction": {"value": False},
        "mixed_precision": {"value": False},
        "tqt": {"value": False},
        "fusion_alignment": {"value": False},
        "blockwise_reconstruction": {"value": False},
    }
    sampled_param = {
        "num_of_bits": {"value": 8},
        "calib_algorithm": {"method": "mse"},
    }

    candidate, num_of_bits = build_quant_setting(
        strategy=strategy,
        sampled_param=sampled_param,
        target="esp32p4",
        layerwise_error={},
        simplified_onnx_path="unused.onnx",
        base_setting=base_setting,
    )

    assert num_of_bits == 8
    assert candidate is not base_setting
    assert candidate.quantize_activation_setting is not base_setting.quantize_activation_setting
    assert candidate.lsq_optimization is True
    assert candidate.quantize_activation_setting.calib_algorithm == "mse"
    assert base_setting.quantize_activation_setting.calib_algorithm == "percentile"


def test_save_experiment_persists_loadable_quantization_setting(tmp_path):
    quant_setting = QuantizationSettingFactory.espdl_setting()
    quant_setting.lsq_optimization = True
    quant_setting.lsq_optimization_setting.steps = 321
    quant_setting.quantize_activation_setting.calib_algorithm = "mse"

    record = save_experiment(
        run_dir=str(tmp_path),
        index=0,
        strategy={"tqt": {"value": False}},
        sampled_param={"tqt": None},
        export_path=str(tmp_path / "model.espdl"),
        runtime_snapshot={"target": "esp32p4"},
        quant_setting=quant_setting,
    )

    setting_path = tmp_path / "0000" / "quant_setting.json"
    loaded = QuantizationSettingFactory.from_json(setting_path.read_text())

    assert "quant_setting.json" in record["files"]
    assert loaded.lsq_optimization is True
    assert loaded.lsq_optimization_setting.steps == 321
    assert loaded.quantize_activation_setting.calib_algorithm == "mse"
    assert loaded.quantize_parameter_setting.clip_visiblity == quant_setting.quantize_parameter_setting.clip_visiblity
