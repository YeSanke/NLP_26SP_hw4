import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt', quiet=True)
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
PREFIX = "translate English to SQL: "
MAX_LENGTH = 512
TOKENIZER_CHECKPOINT = "google-t5/t5-small"


def normalize_sql(sql):
    """
    Normalize SQL string:
    - Uppercase SQL keywords
    - Collapse multiple spaces into one
    """
    keywords = [
        'SELECT', 'DISTINCT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT',
        'IN', 'LIKE', 'IS', 'NULL', 'JOIN', 'ON', 'AS', 'BY',
        'ORDER', 'GROUP', 'HAVING', 'LIMIT', 'UNION', 'ALL'
    ]
    for kw in keywords:
        sql = re.sub(rf'\b{kw}\b', kw, sql, flags=re.IGNORECASE)
    sql = re.sub(r'\s+', ' ', sql).strip()
    return sql


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        """
        Dataset for T5 Text-to-SQL fine-tuning.

        - Encoder input : PREFIX + natural language query (tokenized, truncated to MAX_LENGTH)
        - Decoder input : <extra_id_0> + sql_tokens[:-1]   (teacher forcing input)
        - Decoder target: sql_tokens                        (includes EOS)
        - For test split: only encoder inputs are available
        """
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained(TOKENIZER_CHECKPOINT)
        # <extra_id_0> as BOS token for the decoder
        self.bos_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_lines = load_lines(nl_path)

        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql_lines = load_lines(sql_path)
            sql_lines = [normalize_sql(sql) for sql in sql_lines]
        else:
            sql_lines = [None] * len(nl_lines)

        data = []
        for nl, sql in zip(nl_lines, sql_lines):
            # ── Encoder ───────────────────────────────────────────────────────
            enc = tokenizer(
                PREFIX + nl,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoder_ids  = enc["input_ids"].squeeze(0)       # [T]
            encoder_mask = enc["attention_mask"].squeeze(0)  # [T]

            # ── Decoder ───────────────────────────────────────────────────────
            bos = torch.tensor([self.bos_id], dtype=torch.long)

            if split != "test":
                dec = tokenizer(
                    sql,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                sql_ids = dec["input_ids"].squeeze(0)  # [T'], ends with EOS

                # decoder_input  : [BOS, tok_1, tok_2, ..., tok_{T'-1}]
                # decoder_target : [tok_1, tok_2, ..., tok_{T'-1}, EOS]
                decoder_input  = torch.cat([bos, sql_ids[:-1]])
                decoder_target = sql_ids
            else:
                # Test: only BOS is needed to kick off generation
                decoder_input  = bos
                decoder_target = bos  # placeholder, not used

            data.append((encoder_ids, encoder_mask, decoder_input, decoder_target, bos))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # returns: (encoder_ids, encoder_mask, decoder_input, decoder_target, initial_decoder_input)
        return self.data[idx]


def normal_collate_fn(batch):
    """
    Dynamic padding for train / dev splits.

    Returns:
        encoder_ids       : [B, T]   — padded encoder input ids
        encoder_mask      : [B, T]   — encoder attention mask (0 on pad)
        decoder_inputs    : [B, T']  — padded decoder input ids
        decoder_targets   : [B, T']  — padded decoder target ids (loss computed on non-PAD)
        initial_dec_input : [B, 1]   — BOS token for each sample (used in eval generation)
    """
    encoder_ids_list, encoder_mask_list, dec_in_list, dec_tgt_list, bos_list = zip(*batch)

    # Pad encoder sequences
    encoder_ids  = pad_sequence(encoder_ids_list,  batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask_list, batch_first=True, padding_value=0)

    # Pad decoder sequences
    decoder_inputs  = pad_sequence(dec_in_list,  batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(dec_tgt_list, batch_first=True, padding_value=PAD_IDX)

    # Stack BOS tokens: [B, 1]
    initial_dec_input = torch.stack(bos_list, dim=0)  # each bos is shape [1]

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_dec_input


def test_collate_fn(batch):
    """
    Dynamic padding for test split (no decoder targets).

    Returns:
        encoder_ids       : [B, T]
        encoder_mask      : [B, T]
        initial_dec_input : [B, 1]
    """
    encoder_ids_list, encoder_mask_list, _, _, bos_list = zip(*batch)

    encoder_ids       = pad_sequence(encoder_ids_list,  batch_first=True, padding_value=PAD_IDX)
    encoder_mask      = pad_sequence(encoder_mask_list, batch_first=True, padding_value=0)
    initial_dec_input = torch.stack(bos_list, dim=0)

    return encoder_ids, encoder_mask, initial_dec_input


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader   = get_dataloader(test_batch_size, "dev")
    test_loader  = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x   = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y   = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x  = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x
