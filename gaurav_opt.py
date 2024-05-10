import logging
from typing import Union

import torch
from torch import nn
from transformers import add_start_docstrings
from transformers.pipelines import Pipeline
from transformers.utils import is_ipex_available

from optimum.optimum.exporters.tasks import TasksManager
from optimum_intel.optimum.intel.generation.modeling import jit_trace
from optimum_intel.optimum.intel.ipex.modeling_base import (
    IPEXModel,
    IPEXModelForCausalLM,
    IPEXModelForMaskedLM,
    IPEXModelForSequenceClassification,
    IPEXModelForTokenClassification,
    IPEXModelForQuestionAnswering,
)

import intel_extension_for_pytorch as ipex

_HEAD_TO_AUTOMODELS = {
    "text-generation": "IPEXModelForCausalLM",
    "text-classification": "IPEXModelForSequenceClassification",
    "token-classification": "IPEXModelForTokenClassification",
    "question-answering": "IPEXModelForQuestionAnswering",
}

if is_ipex_available():
    import intel_extension_for_pytorch as ipex


class inference_mode:

    def __init__(
        self,
        model: Union[nn.Module, Pipeline],
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        """
        Args:
            model (`torch.nn.Module` or `transformers.Pipeline`):
                The model or pipeline instance to optimize.
            dtype (`torch.dtype = torch.float32`), *optional*):
                The data type used to do the computation.
                Acceptable type are `torch.float32` (default) and `torch.bfloat16`.
                Please note `torch.bfloat16` requires `avx512_bf16` instructions set as present on
                4th Generation of Intel Xeon Scalable CPUs (Sapphire Rapids).
            jit (`boolean = False`, *optional*):
                Enable jit to accelerate inference speed
        """

        self._model = model
        self._dtype = dtype
        self._original = None
        self._jit = True
        self._original = self._model.model if isinstance(self._model, Pipeline) else self._model
        
    def get_dummy_inputs(self):
            if self._jit:
                        use_cache = getattr(self._original.config, "use_cache", False)
                        task = (
                            self._model.task
                            if isinstance(self._model, Pipeline)
                            else TasksManager._infer_task_from_model_or_model_class(self._model)
                        )
                        if task in _HEAD_TO_AUTOMODELS:
                            model, dummy_inputs = jit_trace(self._model, task, use_cache)
                            print("_HEAD_TO_AUTOMODELS[task]",_HEAD_TO_AUTOMODELS[task])
                            auto_model_class = eval(_HEAD_TO_AUTOMODELS[task])
                            print("auto_model_class",auto_model_class)
                            #model = auto_model_class(model, self._original.config, use_cache=use_cache)
                            #self._model = model
                            
            return dummy_inputs