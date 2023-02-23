from .base import EvaluationTask
import torch


class NiaSummarizationTask(EvaluationTask):

    def prepare_dataset(self):
        self.dataset = load_dataset("heegyu/nia_summary")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        with self.accelerator.local_main_process_first():
            self.mapped_dataset = self.dataset.map(
                self._encode_data, remove_columns=self.dataset["train"].column_names, 
            )

        return self.mapped_dataset


    def evaluation_step(self, batch):
        ppl = torch.exp(
            model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            ).loss
        )

        return {
            "ppl": ppl,
        }