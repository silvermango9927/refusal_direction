from pipeline.model_utils.model_base import ModelBase


def select_model_family(model_path: str) -> str:
    """Resolve a model path to a family key. Pure function so routing is unit-testable
    without loading weights. Order matters: more specific matches come first."""
    p = model_path.lower()

    # Qwen2 / Qwen2.5 / Qwen1.5 use the standard HF Qwen2 arch (model.model.layers) and
    # must NOT go through QwenModel, which is hardcoded for the original Qwen-1
    # (model.transformer.h, tokenizer.eod_id, Qwen-1 refusal token ids).
    if 'qwen2' in p or 'qwen1.5' in p or 'qwen-2' in p:
        return 'qwen2'
    if 'qwen' in p:
        return 'qwen'

    # Llama-3 / Llama-3.1 and Llama-3-derived models (e.g. GovTech SEA-LION v2/v2.1,
    # whose path is "...llama3-8b-sea-lion..." with NO hyphen in "llama3") use the
    # Llama-3 chat template. Match these BEFORE the generic 'llama' (Llama-2) branch,
    # otherwise SEA-LION silently falls through to the Llama-2 [INST] template.
    if 'llama-3' in p or 'llama3' in p or 'llama_3' in p or 'sea-lion' in p or 'sealion' in p:
        return 'llama3'
    if 'llama' in p:
        return 'llama2'

    if 'gemma' in p:
        return 'gemma'
    if 'yi' in p:
        return 'yi'

    raise ValueError(f"Unknown model family: {model_path}")


def construct_model_base(model_path: str) -> ModelBase:
    family = select_model_family(model_path)

    if family == 'qwen2':
        from pipeline.model_utils.qwen2_model import Qwen2Model
        return Qwen2Model(model_path)
    if family == 'qwen':
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)
    if family == 'llama3':
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    if family == 'llama2':
        from pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path)
    if family == 'gemma':
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path)
    if family == 'yi':
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path)

    raise ValueError(f"Unknown model family: {model_path}")
