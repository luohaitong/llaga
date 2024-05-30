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

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llaga_arch import LlagaMetaModel, LlagaMetaForCausalLM
from utils.constants import IGNORE_INDEX


class LlagaConfig(LlamaConfig):
    model_type = "llaga"


class LlagaLlamaModel(LlagaMetaModel, LlamaModel):
    config_class = LlagaConfig

    def __init__(self, config: LlamaConfig):
        super(LlagaLlamaModel, self).__init__(config)
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.post_init()

class LlagaLlamaForCausalLM_fix(LlamaForCausalLM, LlagaMetaForCausalLM):
    config_class = LlagaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlagaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids_desc: torch.LongTensor = None,
        labels_desc: Optional[torch.LongTensor] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph: Optional[torch.FloatTensor] = None,
        graph_emb: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if 1:
            #prepare for interpretation
            if input_ids_desc is not None:
                desc_ids = self.model.generate(
                    input_ids,
                    do_sample=True,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)
                input_token_len = input_ids.shape[1]
                desc_embds = self.get_model().embed_tokens(desc_ids[:, input_token_len:])

                #concat input and interpretation
                labels_desc = torch.full((desc_embds.shape[0],desc_embds.shape[1]),IGNORE_INDEX)
                attention_mask_desc = torch.full((desc_embds.shape[0],desc_embds.shape[1]),True)

        
        #prepare for input and labels
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, graph, graph_emb)
        inputs_embeds = torch.cat([desc_embds, inputs_embeds], dim=1)
        labels = torch.cat([labels_desc, labels],dim=1)
        attention_mask = torch.cat([attention_mask_desc, attention_mask],dim=1) 

        if 0:
            #prepare for interpretation
            if input_ids_desc is not None:
                input_ids_desc, attention_mask_desc, past_key_values_desc, inputs_embeds_desc, labels_desc = self.prepare_inputs_labels_for_multimodal(input_ids_desc, None, past_key_values, labels_desc, graph, graph_emb)
                print("labels desc:", labels_desc)
                outputs_desc = self.model(
                    input_ids=input_ids_desc,
                    attention_mask=attention_mask_desc,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds_desc,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )

                #concat input and interpretation
                labels_desc.fill_(IGNORE_INDEX)
                print("inputs embes before shape:", inputs_embeds.shape)
                inputs_embeds = torch.cat([outputs_desc[0], inputs_embeds], dim=1)
                labels = torch.cat([labels_desc,labels],dim=1)
                #attention mask尺寸也不对

        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph": kwargs.get("graph", None),
                "graph_emb": kwargs.get("graph_emb", None),
            }
        )
        return model_inputs

class LlagaLlamaForCausalLM(LlamaForCausalLM, LlagaMetaForCausalLM):
    config_class = LlagaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlagaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        induced_graph: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, graph, graph_emb)
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal_GNN(input_ids, attention_mask, past_key_values, labels, induced_graph)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            #past_key_values=outputs.past_key_values,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "induced_graph": kwargs.get("induced_graph", None)
            }
        )
        print("model_inpus:",model_inputs['input_ids'])
        return model_inputs
    
class DescriptionLlagaLlamaForCausalLM(LlamaForCausalLM, LlagaMetaForCausalLM):
    config_class = LlagaConfig

    def __init__(self, config):
        self.model = LlagaLlamaForCausalLM(config)

    def get_model(self):
        return self.model.model
    
    def forward(
        self,
        input_ids_desc: torch.LongTensor = None,
        labels_desc: Optional[torch.LongTensor] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph: Optional[torch.FloatTensor] = None,
        graph_emb: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        desc_ids = self.model.generate(
            inputs_ids=input_ids_desc,
            labels=None,
            graph_emb=graph_emb.half().cuda(),
            graph=graph.cuda(),
            do_sample=True,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)
        input_token_len = input_ids.shape[1]
        desc_ids = desc_ids[:, input_token_len:]
        #desc_embds = self.get_model().embed_tokens(desc_ids[:, input_token_len:])

        #concat input and interpretation
        input_ids = torch.cat([desc_ids, input_ids],dim=1)
        labels_desc = torch.full((desc_ids.shape[0],desc_ids.shape[1]),IGNORE_INDEX)
        labels = torch.cat([labels_desc, labels],dim=1)

        return self.model(
            input_ids=input_ids,
            labels=labels,
            graph_emb=graph_emb,
            graph=graph)
    
AutoConfig.register("llaga", LlagaConfig)
AutoModelForCausalLM.register(LlagaConfig, LlagaLlamaForCausalLM)
#AutoModelForCausalLM.register(LlagaConfig, DescriptionLlagaLlamaForCausalLM)
