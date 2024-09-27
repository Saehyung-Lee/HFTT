from transformers import AutoProcessor, BlipForImageTextRetrieval
import torch
from typing import Dict, Callable, Optional, Any, Tuple, Union
import clip
from torch import nn

class BlipForRetrieval(BlipForImageTextRetrieval):
    def set_ood_classifier(self, num_ood_classes):
        embed_dim = self.vision_proj.weight.size(0)
        self.ood_classifier_ = nn.Parameter(torch.randn(size=(embed_dim, num_ood_classes), dtype=torch.float))

    def encode_text(self,
                          input_ids: torch.LongTensor,
                          attention_mask: Optional[torch.LongTensor] = None,
                          return_dict: Optional[bool] = None,
                          ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

        text_feat = self.text_proj(question_embeds[:, 0, :])

        return text_feat

    def encode_image(
            self,
            pixel_values: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        with torch.no_grad():
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            image_embeds = vision_outputs[0]

            image_feat = self.vision_proj(image_embeds[:, 0, :])

        return image_feat

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 

def get_model(args):
    if args.model == 'clip-base':
        ''' temp = 0.01 '''
        model, processor = clip.load("ViT-B/16", device='cpu', args=args)
        convert_models_to_fp32(model)
    elif args.model == 'clip-large':
        ''' temp = 0.01 '''
        model, processor = clip.load("ViT-L/14", device='cpu', args=args)
        convert_models_to_fp32(model)
    elif args.model == 'blip-base':
        ''' temp = 0.0174 '''
        model = BlipForRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    elif args.model == 'blip-large':
        ''' temp = 0.0222 '''
        model = BlipForRetrieval.from_pretrained("Salesforce/blip-itm-large-coco")
        processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
    return model, processor
