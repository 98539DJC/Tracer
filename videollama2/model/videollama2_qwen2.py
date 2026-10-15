# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import os
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .videollama2_arch import Videollama2MetaModel, Videollama2MetaForCausalLM

_MEM_DEBUG = os.environ.get('TRACER_MEM_DEBUG', '0') == '1'


class Videollama2Qwen2Config(Qwen2Config):
    model_type = "videollama2_qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "videollama2_qwen2"


class Videollama2Qwen2Model(Videollama2MetaModel, Qwen2Model):
    config_class = Videollama2Qwen2Config

    def __init__(self, config: Videollama2Qwen2Config):
        super(Videollama2Qwen2Model, self).__init__(config)


class Videollama2Qwen2ForCausalLM(Qwen2ForCausalLM, Videollama2MetaForCausalLM):
    config_class = Videollama2Qwen2Config

    def __init__(self, config, **kwargs):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = Videollama2Qwen2Model(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        wordweights: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        loss_data = None
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                loss_data
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
        if labels is not None and wordweights is not None:
            # Align every sample's wordweights with its own label positions.  Samples
            # without a weight file are filled with the mean of the weights present in
            # this batch so the scale of L_cap stays consistent.
            _missing, _present_vals = [], []
            for index, label in enumerate(labels):
                wordweight = wordweights[index]
                padded_wordweight = torch.zeros_like(label, dtype=torch.float32)
                valid_indices = torch.where(label != -100)[0]
                num_valid = valid_indices.size(0)
                if wordweight is not None and num_valid > 0:
                    wordweight_tensor = torch.tensor(wordweight, dtype=torch.float32)
                    wordweight_len = len(wordweight)
                    min_len = min(wordweight_len, num_valid)
                    if min_len > 0:
                        padded_wordweight[valid_indices[:min_len]] = wordweight_tensor[:min_len]
                    remaining = num_valid - min_len
                    if remaining > 0:
                        mean_value = wordweight_tensor.mean() if wordweight_len > 0 else 0.0
                        padded_wordweight[valid_indices[min_len:]] = mean_value
                    _present_vals.append(padded_wordweight[valid_indices])
                elif wordweight is None:
                    _missing.append((index, valid_indices))
                wordweights[index] = padded_wordweight
            if _missing:
                _fill = (torch.cat(_present_vals).mean() if _present_vals
                         else torch.tensor(1.0, device=labels.device))
                for _i, _vi in _missing:
                    if _vi.numel():
                        wordweights[_i][_vi] = _fill
            wordweights = torch.stack(wordweights)
            if loss_data is not None:
                loss_data['wordweights'] = wordweights
            
        
        # 手动前向：不用 super().forward，避免 transformers Qwen2ForCausalLM.forward
        # 里的 `logits = logits.float()` 把 bf16 logits 转回 fp32（显存翻倍、OOM 元凶）。
        # 这里 logits 保持 bf16。
        if _MEM_DEBUG:
            print(f'[MEMDBG] before LLM: {torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = model_outputs[0]
        if _MEM_DEBUG:
            print(f'[MEMDBG] after LLM: {torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)
        logits = self.lm_head(hidden_states)
        if _MEM_DEBUG:
            print(f'[MEMDBG] after logits: {torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flat_logits = shift_logits.view(-1, self.config.vocab_size)
            flat_labels = shift_labels.view(-1)

            # ---- word-weighted language-modelling loss (L_cap) ----
            # `wordweights` is (b, L) aligned with `labels` (built above), so it
            # takes the same [1:] shift.  Each GT token's cross-entropy is
            # multiplied by that token's weight and SUMMED (no normalisation).
            # NOTE: a sum scales with the number of tokens in the batch, so
            # changing the batch size changes the effective learning rate.
            use_ww = (
                getattr(self.config, 'use_wordweight_loss', True)
                and isinstance(wordweights, torch.Tensor)
                and wordweights.dim() == 2
                and wordweights.shape[0] == labels.shape[0]
                and wordweights.shape[1] == labels.shape[1]
            )
            # Chunked CE: the naive call upcasts the entire (b*L, vocab) logits
            # tensor to fp32 inside the kernel -- 6.4 GiB at b=16 -- which is what
            # limits the batch size.  Slicing caps the workspace at
            # CE_CHUNK * vocab * 4 bytes and is numerically identical.
            _ce_chunk = int(os.environ.get('CE_CHUNK', '') or
                            getattr(self.config, 'ce_chunk_size', 0) or 1024)

            def _per_token_ce(logits2d, labels1d):
                if _ce_chunk <= 0 or logits2d.shape[0] <= _ce_chunk:
                    return nn.CrossEntropyLoss(reduction='none')(logits2d, labels1d)
                parts = []
                for _s in range(0, logits2d.shape[0], _ce_chunk):
                    parts.append(nn.CrossEntropyLoss(reduction='none')(
                        logits2d[_s:_s + _ce_chunk], labels1d[_s:_s + _ce_chunk]))
                return torch.cat(parts)

            if use_ww:
                shift_weights = wordweights[..., 1:].contiguous().to(flat_logits.dtype).view(-1)
                valid = flat_labels.ne(-100)
                weights = torch.where(valid, shift_weights, torch.zeros_like(shift_weights))
                per_token = _per_token_ce(flat_logits, flat_labels)
                per_token = torch.nan_to_num(per_token, nan=0.0, posinf=0.0, neginf=0.0)
                if float(weights.sum()) > 0.0:
                    _norm = getattr(self.config, 'wordweight_normalize', 'sum')
                    if _norm == 'mean':          # weighted mean
                        loss = (per_token * weights).sum() / weights.sum().clamp(min=1e-6)
                    elif _norm == 'count':       # per valid token
                        loss = (per_token * weights).sum() / valid.sum().clamp(min=1)
                    else:                        # 'sum' (default): plain weighted sum
                        loss = (per_token * weights).sum()
                elif bool(valid.any()):
                    loss = per_token[valid].mean()
                else:
                    loss = per_token.sum() * 0.0
                if os.environ.get('WORDWEIGHT_DEBUG') == '1':
                    with torch.no_grad():
                        _plain = (per_token[valid].mean() if bool(valid.any())
                                  else per_token.sum() * 0.0)
                        _wsum = float(weights.sum())
                        _n = int(valid.sum())
                        print('[WWDBG] labels=%s ww=%s n_valid=%d w_sum=%.6f '
                              'plain_ce=%.6f weighted_ce=%.6f norm=%s'
                              % (tuple(labels.shape), tuple(wordweights.shape), _n, _wsum,
                                 float(_plain), float(loss),
                                 getattr(self.config, 'wordweight_normalize', 'sum')),
                              flush=True)
            else:
                # plain CE, still chunked so the fp32 upcast does not blow up
                _valid = flat_labels.ne(-100)
                _pt = _per_token_ce(flat_logits, flat_labels)
                _pt = torch.nan_to_num(_pt, nan=0.0, posinf=0.0, neginf=0.0)
                if bool(_valid.any()):
                    loss = _pt.sum() / _valid.sum().clamp(min=1)
                else:
                    loss = _pt.sum() * 0.0

        if return_dict is False:
            output = (logits,) + model_outputs[1:]
            outputs = ((loss,) + output) if loss is not None else output
        else:
            outputs = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=model_outputs.past_key_values,
                hidden_states=model_outputs.hidden_states,
                attentions=model_outputs.attentions,
            )
        if loss_data is not None:
            outputs["loss_data"] = loss_data
        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=images
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("videollama2_qwen2", Videollama2Qwen2Config)
AutoModelForCausalLM.register(Videollama2Qwen2Config, Videollama2Qwen2ForCausalLM)
