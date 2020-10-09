import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import logging


class WeightedClassifier(object):
    def __init__(self, model_name: str, hyper_dict: dict, device_name: str = "cuda") -> None:
        assert torch.cuda.is_available(), "cuda required"
        self.device = torch.device(device_name)
        self.device_name = device_name

        self.hyper_dict = hyper_dict

        self.label_dict = {
            "F": 0,
            "T": 1,
            "R": 2
        }
        logging.info(f"label_dict: {self.label_dict}")
        self.inv_label_dict = {value: key for key, value in self.label_dict.items()}

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(self.label_dict))
        self.model.cuda(self.device_name)
        self.check_count = 0

    def tokenize_encode(self, text: str) -> list:
        tokens = self.tokenizer.tokenize(text)
        encoded_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return encoded_tokens

    @staticmethod
    def pad_sequences(sequences: list, max_len: int) -> torch.Tensor:
        out_dims = (len(sequences), max_len)
        out_tensor = torch.zeros(*out_dims, dtype=torch.long)

        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            if length <= max_len:
                out_tensor[i, :length, ...] = tensor
            else:
                out_tensor[i, :length, ...] = tensor[:max_len]

        return out_tensor

    @staticmethod
    def add_special_token(sentence, index, special_token="[MASK]"):
        split_sent = sentence.split()
        split_sent.insert(special_token, index)
        return " ".join(split_sent)

    def read_data(self, df: pd.DataFrame, for_test: bool = False):
        input_ids = []
        token_type_ids = []

        count_overlong = 0
        for first_sentence, second_sentence, first_index, second_index in zip(df.sent_1.values, df.sent_2.values,
                                                                              df.index_1.values, df.index_2.values):

            assert type(first_sentence) == str,  f"first sentence not string: {first_sentence} | {second_sentence}"
            assert type(second_sentence) == str, f"second sentence not string: {first_sentence} | {second_sentence}"

            if self.hyper_dict["add_special"]:
                first_sentence = self.add_special_token(first_sentence, int(first_index))
                second_sentence = self.add_special_token(second_sentence, int(second_index))

            first_ids = self.tokenize_encode("[CLS] " + first_sentence + " [SEP]")
            first_types = [0 for _ in first_ids]
            second_ids = self.tokenize_encode(second_sentence + " [SEP]")
            second_types = [1 for _ in second_ids]

            if len(first_ids+second_ids) > self.hyper_dict["max_len"]:
                count_overlong += 1

            input_ids.append(torch.tensor(first_ids + second_ids))
            token_type_ids.append(torch.tensor(first_types + second_types))

        logging.info(f"Found {count_overlong} definition pairs exceeding max_len")

        input_ids = self.pad_sequences(input_ids, self.hyper_dict["max_len"])
        token_type_ids = self.pad_sequences(token_type_ids, self.hyper_dict["max_len"])

        attention_masks = []

        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        attention_masks = torch.tensor(attention_masks, dtype=torch.long)

        assert type(input_ids) == torch.Tensor, f"Wrong type: {type(input_ids)} for input_ids. Should be Tensor"
        assert type(token_type_ids) == torch.Tensor, \
            f"Wrong type: {type(token_type_ids)} for token_type_ids. Should be Tensor"
        assert type(attention_masks) == torch.Tensor, \
            f"Wrong type: {type(attention_masks)} for attention_masks. Should be Tensor"

        if for_test:
            tensor_data = TensorDataset(input_ids, token_type_ids, attention_masks)
        else:
            labels = [self.label_dict[label] for label in df.label.values]
            labels = torch.tensor(labels, dtype=torch.long)
            tensor_data = TensorDataset(input_ids, token_type_ids, attention_masks, labels)

        assert df.shape[0] == input_ids.shape[0], "length of DataFrame and length of input do not match"

        return tensor_data, df

    def convert_labels(self, label_tensor):
        converted_labels = []
        base = [0, 0, 0]
        for label in label_tensor:
            converted_labels.append(base[:label] + [1] + base[label + 1:])

        return torch.tensor(converted_labels, dtype=torch.float, device=self.device)

    def check_seg_ids(self, seg_id_batch) -> bool:
        change_count = 0
        expected = 0

        for seg_id in seg_id_batch[0]:
            if seg_id == expected:
                continue
            elif (seg_id == 1) and (change_count == 0):
                expected = 1
                change_count += 1
            elif (seg_id == 0) and (change_count == 1):
                expected = 0
                change_count += 1
            else:
                logging.warning(seg_id_batch[0])
                return False

        if change_count == 0:
            self.check_count += 1

        return True

    def batch_train(self, batch, optimizer, scheduler):
        pos_weight = torch.tensor(self.hyper_dict["pos_weight"], dtype=torch.long, device=self.device)

        batch = tuple(t.to(self.device) for t in batch)
        b_input_ids, b_type_ids, b_input_mask, b_labels = batch
        assert self.check_seg_ids(b_type_ids), "Segment IDs not in correct format"

        optimizer.zero_grad()
        outputs = self.model(b_input_ids, token_type_ids=b_type_ids, attention_mask=b_input_mask)
        logits = outputs[0]
        converted_labels = self.convert_labels(b_labels)

        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_fct.to(self.device)
        loss = loss_fct(
            logits, converted_labels
        )

        return_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_dict["max_grad_norm"])  # ordering?
        optimizer.step()
        scheduler.step()  # ordering assumes pytorch >= 1.1.0
        return return_loss

    def train_model(self, training_data: pd.DataFrame):
        logging.info(f"starting to train model")
        encoded_data, df = self.read_data(training_data)

        assert training_data is df

        sampler = RandomSampler(encoded_data)
        dataloader = DataLoader(encoded_data, sampler=sampler, batch_size=self.hyper_dict["batch_size"])

        steps_per_epoch = df.shape[0] // self.hyper_dict["batch_size"]
        num_total_steps = steps_per_epoch * self.hyper_dict["epochs"]

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': self.hyper_dict["decay_rate"]},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hyper_dict["learning_rate"], correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hyper_dict["warmup_steps"],
                                                    num_training_steps=num_total_steps)

        self.model.train()

        for _ in range(self.hyper_dict["epochs"]):
            logging.info("starting epoch")

            training_loss = [self.batch_train(batch, optimizer, scheduler) for batch in dataloader]

            logging.info(f"Training loss: {sum(training_loss)/len(training_loss)}")

    def eval_model(self, eval_data: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"starting to test model")

        encoded_data, df = self.read_data(eval_data, for_test=True)
        sampler = SequentialSampler(encoded_data)
        dataloader = DataLoader(encoded_data, sampler=sampler, batch_size=self.hyper_dict["batch_size"])

        self.model.eval()

        predictions = []

        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_type_ids, b_input_mask = batch
            assert self.check_seg_ids(b_type_ids), "Segment IDs not in correct format"

            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=b_type_ids, attention_mask=b_input_mask)[0]

            logits = logits.detach().cpu().numpy()
            b_predictions = np.argmax(logits, axis=1).flatten()

            predictions.extend([int(prediction) for prediction in b_predictions])

        predicted_labels = [self.inv_label_dict[label] for label in predictions]
        logging.warning(f"Not a single part of second definition seen during training in {self.check_count} cases.")

        return df.assign(label=predicted_labels)  # returns copy with new labels
