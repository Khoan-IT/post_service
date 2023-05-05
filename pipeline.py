import os
import yaml
import torch
import json

import numpy as np

from typing import List, Dict, Union
from architecture import JointPhoBERT
from transformers import (
    AutoTokenizer,
    RobertaConfig
)


class IntentSlotModel:
    def __init__(self, model_path: str, **kwargs):
        self.config = self._get_config(model_path)
        self._init_model()
        self._init_tokenizer()
        
        
    def _get_config(self, model_path: str) -> Dict[str, Dict[str, Union[str, bool, int]]]:
        with open(os.path.join(model_path, 'config.yaml'), 'r') as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return config
    
    
    def _init_model(self) -> None:
        if not os.path.exists(os.path.join(self.config['checkpoint']['checkpoint_dir'])):
            raise Exception("Checkpoint doesn't exists!")
        try:
            self.intent_label_lst = [label.strip() for label in open(os.path.join(self.config['data']['data_dir'], self.config['data']['intent_label_name']))]
            self.slot_label_lst = [slot.strip() for slot in open(os.path.join(self.config['data']['data_dir'], self.config['data']['slot_label_name']))]
            self.args_model = torch.load(os.path.join(self.config['checkpoint']['checkpoint_dir'], "training_args.bin"))
            
            self.model = JointPhoBERT.from_pretrained(
                                self.config['checkpoint']['checkpoint_dir'], 
                                args=self.args_model, 
                                intent_label_lst=self.intent_label_lst, 
                                slot_label_lst=self.slot_label_lst
                            )
            if self.config['device']['cuda']:
                self.model.to('cuda')
            self.model.eval()
        except Exception:
            raise Exception("Some checkpoint files might be missing...")
    
    
    def _init_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.args_model.model_name_or_path)
        # Setting based on the current model type
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.unk_token = self.tokenizer.unk_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pad_token_label_id = self.args_model.ignore_index
        
    
    def _convert_input_to_tensor(self, words: List[str]) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        token_type_ids = []
        slot_label_mask = []

        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [self.unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([self.pad_token_label_id + 1] + [self.pad_token_label_id] * (len(word_tokens) - 1))
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > self.args_model.max_seq_len - special_tokens_count:
            tokens = tokens[: (self.args_model.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[: (self.args_model.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [self.sep_token]
        token_type_ids = [self.config['tokenizer']['sequence_a_segment_id']] * len(tokens)
        slot_label_mask += [self.pad_token_label_id]

        # Add [CLS] token
        tokens = [self.cls_token] + tokens
        token_type_ids = [self.config['tokenizer']['cls_token_segment_id']] + token_type_ids
        slot_label_mask = [self.pad_token_label_id] + slot_label_mask

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if self.config['tokenizer']['mask_padding_with_zero'] else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.args_model.max_seq_len - len(input_ids)
        input_ids = input_ids + ([self.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if self.config['tokenizer']['mask_padding_with_zero'] else 1] * padding_length)
        token_type_ids = token_type_ids + ([self.config['tokenizer']['sequence_a_segment_id']] * padding_length)
        slot_label_mask = slot_label_mask + ([self.pad_token_label_id] * padding_length)

        # Change to Tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)
        slot_label_mask = torch.tensor([slot_label_mask], dtype=torch.long)
        
        if self.config['device']['cuda']:
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            token_type_ids = token_type_ids.to('cuda')
            slot_label_mask = slot_label_mask.to('cuda')

        input_data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'slot_label_mask': slot_label_mask
        }

        return input_data
    
    
    def _predict(self, input_data: Dict[str, torch.Tensor], words: List[str])-> str:
        all_slot_label_mask = None
        intent_preds = None
        slot_preds = None
        
        with torch.no_grad():
            inputs = {
                "input_ids": input_data['input_ids'],
                "attention_mask": input_data['attention_mask'],
                "intent_label_ids": None,
                "slot_labels_ids": None,
            }
            if self.args_model.model_type != "distilbert":
                inputs["token_type_ids"] = input_data['token_type_ids'].to('cuda') if self.config['device']['cuda'] else input_data['token_type_ids']
                
            outputs = self.model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]
            
            # Intent Prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()

            # Slot prediction
            if slot_preds is None:
                if self.args_model.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                slot_label_mask = input_data['slot_label_mask'].detach().cpu().numpy()

        intent_preds = np.argmax(intent_preds, axis=1)

        if not self.args_model.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
            
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        slot_preds_list = []

        for i in range(slot_preds.shape[0]):
            for j in range(slot_preds.shape[1]):
                if slot_label_mask[i, j] != self.pad_token_label_id:
                    slot_preds_list.append(slot_label_map[slot_preds[i][j]])
        print(slot_preds_list)
        slots = {}
        current_slot = slot_preds_list[0]
        value_slot = ''
        len_sentence = len(words)
        for i, (word, pred_slot) in enumerate(zip(words, slot_preds_list)):
            if current_slot != pred_slot:
                if current_slot != 'O':
                    if current_slot not in slots:
                        slots[current_slot] = value_slot.strip()
                    else:
                        slots[current_slot] = slots[current_slot] + ', ' + value_slot.strip()
                    value_slot = ''
                current_slot = pred_slot
                    
            if current_slot == pred_slot and pred_slot != 'O':
                value_slot = value_slot + ' ' + word
            
            if i == len_sentence - 1 and current_slot != 'O':
                if current_slot not in slots:
                    slots[current_slot] = value_slot.strip()
                else:
                    slots[current_slot] = slots[current_slot] + ', ' + value_slot.strip()
                value_slot = ''

        return slots

              
    def __call__(self, input_message: str) -> str:
        posts = json.loads(input_message)
        results = []
        for post in posts:
            res_post = {"id": post['id']}
            words = post['content'].strip().split()
            res_post['activity'] = self._predict(self._convert_input_to_tensor(words), words)
            results.append(res_post)
        
        return json.dumps(results)
        

    
    


    