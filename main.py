from datasets import load_metric
from dataset import CustomDataset
from utils import postprocess_text, load_tokenizer
from transformers import EncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from log import log_progress
from config import Config
import numpy as np
import os
import evaluate
import torch
torch.cuda.empty_cache()

class Manager():
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg

        print("Loading Tokenizer...")
        self.get_tokenizer()

        print("Loading Model...")
        self.get_model()

        print("Loading Metric...")
        self.bleu_metric = evaluate.load("sacrebleu")

        print("Check Save Model Path")
        if not os.path.exists(self.cfg.ckpt_dir):
            os.mkdir(self.cfg.ckpt_dir)

        if is_train:
            # Load dataloaders
            print("Loading Dataset...")
            self.train_dataset = CustomDataset(self.cfg, data_type="train")
            self.valid_dataset = CustomDataset(self.cfg, data_type="validation")

        print("Setting finished.")

    def get_tokenizer(self):
        if self.cfg.load_model_from_path:
            self.cfg.src_tokenizer = load_tokenizer(self.cfg.ckpt_dir)
            self.cfg.tgt_tokenizer = load_tokenizer(self.cfg.ckpt_dir)
        else:
            self.cfg.src_tokenizer = load_tokenizer(self.cfg.src_model_name)
            self.cfg.tgt_tokenizer = load_tokenizer(self.cfg.tgt_model_name)
            if "bert" in self.cfg.tgt_model_name.split('-'):
                self.cfg.add_special_tokens = False
                self.cfg.bos_token_id = self.cfg.tgt_tokenizer.cls_token_id
                self.cfg.eos_token_id = self.cfg.tgt_tokenizer.sep_token_id
                self.cfg.pad_token_id = self.cfg.tgt_tokenizer.pad_token_id
            else:
                self.cfg.add_special_tokens = True
                self.cfg.tgt_tokenizer.add_special_tokens(
                    {
                        "bos_token": "[BOS]",
                        "eos_token": "[EOS]",
                        "pad_token": "[PAD]"
                    }
                )
                self.cfg.bos_token_id = self.cfg.tgt_tokenizer.bos_token_id
                self.cfg.eos_token_id = self.cfg.tgt_tokenizer.eos_token_id
                self.cfg.pad_token_id = self.cfg.tgt_tokenizer.pad_token_id
        self.cfg.src_tokenizer.save_pretrained(
                os.path.join(self.cfg.ckpt_dir, f"{self.cfg.src_lang}_tokenizer_{self.cfg.src_model_name}")
            )

        self.cfg.tgt_tokenizer.save_pretrained(
                os.path.join(self.cfg.ckpt_dir, f"{self.cfg.tgt_lang}_tokenizer_{self.cfg.tgt_model_name}")
            )

    def get_model(self):
        if self.cfg.load_model_from_path:
            save_model_path = os.path.join(self.cfg.ckpt_dir, self.cfg.ckpt_name)
            self.model = EncoderDecoderModel.from_pretrained(save_model_path)
        else:
            self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.cfg.src_model_name,
                self.cfg.tgt_model_name
            )
            self.model.decoder.resize_token_embeddings(len(self.cfg.tgt_tokenizer))
            self.model.config.decoder_start_token_id = self.cfg.bos_token_id
            self.model.config.eos_token_id = self.cfg.eos_token_id
            self.model.config.pad_token_id = self.cfg.pad_token_id
            self.model.config.vocab_size = len(self.cfg.tgt_tokenizer)
            self.model.config.max_length = self.cfg.max_length_decoder
            self.model.config.min_length = self.cfg.min_length_decoder
            self.model.config.no_repeat_ngram_size = 3
            self.model.config.early_stopping = True
            self.model.config.length_penalty = 2.0
            self.model.config.num_beams = self.cfg.beam_size

    def train(self):
        print("Training...")
        if self.cfg.use_eval_steps:
            training_args = Seq2SeqTrainingArguments(
                predict_with_generate=True,
                evaluation_strategy="steps",
                save_strategy='steps',
                save_steps=self.cfg.eval_steps,
                eval_steps=self.cfg.eval_steps,
                output_dir=self.cfg.ckpt_dir,
                per_device_train_batch_size=self.cfg.train_batch_size,
                per_device_eval_batch_size=self.cfg.eval_batch_size,
                learning_rate=self.cfg.learning_rate,
                weight_decay=0.005,
                num_train_epochs=self.cfg.num_train_epochs,
                fp16=True,  # Enable mixed precision training
                gradient_accumulation_steps=10,  # Accumulate gradients over 2 steps
                logging_dir='./logs',
                gradient_checkpointing=True,  # Enable gradient checkpointing
            )
        else:
            training_args = Seq2SeqTrainingArguments(
                predict_with_generate=True,
                evaluation_strategy="epoch",
                save_strategy='epoch',
                output_dir=self.cfg.ckpt_dir,
                per_device_train_batch_size=self.cfg.train_batch_size,
                per_device_eval_batch_size=self.cfg.eval_batch_size,
                learning_rate=self.cfg.learning_rate,
                weight_decay=0.005,
                num_train_epochs=self.cfg.num_train_epochs,
                fp16=True,  # Enable mixed precision training
                gradient_accumulation_steps=10,  # Accumulate gradients over 2 steps
                logging_dir='./logs',
                gradient_checkpointing=True,  # Enable gradient checkpointing
            )

        data_collator = DataCollatorForSeq2Seq(
            self.cfg.tgt_tokenizer,
            model=self.model
        )

        trainer = Seq2SeqTrainer(
            self.model,
            training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            data_collator=data_collator,
            tokenizer=self.cfg.tgt_tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.cfg.tgt_tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.cfg.tgt_tokenizer.pad_token_id)
        decoded_labels = self.cfg.tgt_tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = self.bleu_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )

        result = {"bleu_score": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.cfg.tgt_tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result
    
if __name__ == "__main__":
    cfg = Config('config.yaml')
    manager = Manager(cfg,is_train=True)
    manager.train()
