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

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

import time 
start_time = time.time()
results = model(torch.tensor([[ 6524,  5995, 21406, 23717,  6107, 27901, 24058, 28898,  7686,  9818,
         21261, 20895, 10262, 24293, 10446,  7597],
        [27232, 11686,  7944, 21532,  9522, 24180, 16246, 27213,  2027, 25996,
          6363, 10267,  8071, 26206, 25833,  3596]]),
                torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]))
end_time = time.time()
print("time taken", end_time-start_time)


'''jit = True
use_cache = True

with torch.inference_mode():
            ipex.enable_onednn_fusion(True)

            original = model.model if isinstance(model, Pipeline) else model
                    
            if jit:
                    task = (
                            model.task
                            if isinstance(model, Pipeline)
                            else TasksManager._infer_task_from_model_or_model_class(model)
                        )
                    if task in _HEAD_TO_AUTOMODELS:
                            model = jit_trace(model, task, use_cache)
                            auto_model_class = eval(_HEAD_TO_AUTOMODELS[task])
                            model = auto_model_class(model, original.config, use_cache=use_cache)
'''
  
from gaurav_opt import inference_mode


#with inference_mode(pipe, dtype=torch.bfloat16, jit=True) as opt_pipe:
    #results = opt_pipe("He's a dreadful magician.")
    #print("pipeline results", results)
              
obj = inference_mode(model=model,dtype=torch.bfloat16)
dumpy_tensor = obj.get_dummy_inputs()

model = ipex.optimize(model, dtype=torch.bfloat16, level='O0')
jit_inputs = (dumpy_tensor["input_ids"], dumpy_tensor["attention_mask"])
with torch.cpu.amp.autocast(), torch.no_grad():
    model = torch.jit.trace(model, jit_inputs, strict=False)
    model = torch.jit.freeze(model)
with torch.no_grad():
    y = model(dumpy_tensor["input_ids"], dumpy_tensor["attention_mask"])
    y = model(dumpy_tensor["input_ids"], dumpy_tensor["attention_mask"])
                    
                    
start_time = time.time()
with torch.no_grad():
    y = model(dumpy_tensor["input_ids"], dumpy_tensor["attention_mask"])
end_time = time.time()
print("time taken", end_time-start_time)



