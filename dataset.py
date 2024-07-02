from torch.utils.data import Dataset
from datasets import load_dataset
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, cfg, data_type="train"):
        super().__init__()
        self.cfg = cfg

        self.src_texts, self.tgt_texts = self.read_data(data_type)

        self.src_input_ids, self.src_attention_mask = self.texts_to_sequences(self.src_texts)
        self.tgt_input_ids, self.tgt_attention_mask, self.labels = self.texts_to_sequences(
            self.tgt_texts,
            is_src=False
        )

    def read_data(self, data_type):
        data = load_dataset(
            self.cfg.dataset_name,
            split=data_type
        )
        src_texts = data[self.cfg.src_lang][:100]
        tgt_texts = data[self.cfg.tgt_lang][:100]
        print(len(src_texts),len(tgt_texts))
        return src_texts, tgt_texts

    def texts_to_sequences(self, texts, is_src=True):
        if is_src:
            src_inputs = self.cfg.src_tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.cfg.src_max_len,
                return_tensors='pt'
            )
            return (
                src_inputs.input_ids,
                src_inputs.attention_mask
            )

        else:
            if self.cfg.add_special_tokens:
                texts = [
                    ' '.join([
                        self.cfg.tgt_tokenizer.bos_token,
                        text,
                        self.cfg.tgt_tokenizer.eos_token
                        ])
                    for text in texts
                ]
            tgt_inputs = self.cfg.tgt_tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.cfg.tgt_max_len,
                return_tensors='pt'
            )

            labels = tgt_inputs.input_ids.numpy().tolist()
            labels = [
                [
                    -100 if token_id == self.cfg.tgt_tokenizer.pad_token_id else token_id
                    for token_id in label
                ]
                for label in labels
            ]

            labels = torch.LongTensor(labels)

            return (
                tgt_inputs.input_ids,
                tgt_inputs.attention_mask,
                labels
            )

    def __getitem__(self, idx):
        return {
            "input_ids": self.src_input_ids[idx],
            "attention_mask": self.src_attention_mask[idx],
            "decoder_input_ids": self.tgt_input_ids[idx],
            "decoder_attention_mask": self.tgt_attention_mask[idx],
            "labels": self.labels[idx]
        }

    def __len__(self):
        return np.shape(self.src_input_ids)[0]