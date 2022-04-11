from torch.utils.data import Dataset
from datasets import load_dataset
import json
import torch
from utils import EditBatchSampler, dict_to, scr
import logging
import random
import copy

LOG = logging.getLogger(__name__)


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def filter_text(iterator):
    valid = []
    for text in iterator:
        if len(text.split(' ')) < 50:
            continue
        if not is_ascii(text):
            continue
        valid.append(text)

    return valid


class CFGenDataset(Dataset):
    def __init__(self, split: str, tokenizer, config, edit_path: str,
                 pct: int = 10, max_length: int = 200):
        assert split in ["train", "validation"]

        with open("/home/mengk/mend/data/counterfact/counterfact.json", "r") as f:
            self.edit_samples = json.load(f)
            self.edit_samples = self.edit_samples[-10000:] if split == "train" else self.edit_samples[:1000]

        self.tok = tokenizer
        self.config = config
        self.max_length = max_length

        len_edit = len(self.edit_samples)
        LOG.info(f"Loaded {len_edit} edit samples")

    def edit_generator(self, batch_size, n=None):
        if n is None:
            n = len(self)

        edit_idx = 0
        while True:
            record = self.edit_samples[edit_idx]
            request = record["requested_rewrite"]

            target_new_tok = self.tok(" " + request["target_new"]["str"])["input_ids"]
            target_old_tok = self.tok(" " + request["target_true"]["str"])["input_ids"]

            edit_batch = [request["prompt"].format(request["subject"]) + self.tok.decode(target_new_tok[:-1])]
            loc_batch = [record["neighborhood_prompts"][0] + self.tok.decode(target_old_tok[:-1])]
            para_batch = [record["paraphrase_prompts"][0] + self.tok.decode(target_new_tok[:-1])]

            edit_toks = self.tok(edit_batch, padding=True, return_tensors="pt")
            loc_toks = self.tok(loc_batch, padding=True, return_tensors="pt")
            para_toks = self.tok(para_batch, padding=True, return_tensors="pt")

            # print(edit_batch, edit_toks)
            # print(loc_batch, loc_toks)

            def edit_labels(inp, target_tok):
                ids, amask = inp["input_ids"], inp["attention_mask"]

                ans = []
                for batch_idx in range(ids.size(0)):
                    batch_ids = ids[batch_idx]
                    batch_amask = amask[batch_idx]

                    ret = torch.tensor(-100).repeat(len(batch_ids))
                    pad_factor = batch_amask.sum()
                    for i, el in enumerate(target_tok):
                        ret[pad_factor - len(target_tok) + i] = el

                    ret[ret == -100] = self.tok.eos_token_id
                    # print(self.tok.decode(ret))
                    # print(list(zip([self.tok.decode(z) for z in batch_ids], [self.tok.decode(z) for z in ret])))
                    ans.append(ret)
                # kevin meng goes to    mass   inst   ??
                # -100 -100  -100 mass   inst tech
                # kevin meng ii   goes   to    mass   inst 

                return torch.stack(ans, 0)

            edit_inner = {**edit_toks}
            edit_inner["labels"] = edit_labels(edit_toks, target_new_tok)

            edit_outer = {**para_toks}
            edit_outer["labels"] = edit_labels(para_toks, target_new_tok)

            loc = {**loc_toks}
            loc["labels"] = edit_labels(loc_toks, target_old_tok)

            cond = {**edit_toks}

            batch = {
                "edit_inner": edit_inner,
                "edit_outer": edit_outer,
                "loc": loc,
                "cond": cond
            }

            edit_idx += 1
            yield dict_to(batch, self.config.device)

    def __len__(self):
        return len(self.edit_samples)

    def _check_padding(self, ids):
        if (ids[:, 0] == self.tok.pad_token_id).any():
            raise ValueError("Left-padding not supported for GPT2")

    def get_edit_labels(self, ids):
        self._check_padding(ids)

        labels = ids.clone()
        end_idxs = (labels != self.tok.pad_token_id).sum(-1)
        for batch_idx, end_idx in enumerate(end_idxs):
            labels[batch_idx, :end_idx - self.n_tokens] = -100
        labels[labels == self.tok.pad_token_id] = -100
        return labels

    def get_labels(self, ids):
        self._check_padding(ids)

        return ids.masked_fill(ids == self.tok.pad_token_id, -100)

    def __getitem__(self, idx):
        return self.base_samples[idx]
