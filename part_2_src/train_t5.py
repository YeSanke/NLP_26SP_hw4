import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records, read_queries

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0
TOKENIZER_CHECKPOINT = "google-t5/t5-small"
TOKENIZER = T5TokenizerFast.from_pretrained(TOKENIZER_CHECKPOINT)


def get_args():
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")

    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"])
    parser.add_argument('--num_warmup_epochs', type=int, default=1)
    parser.add_argument('--max_n_epochs', type=int, default=30)
    parser.add_argument('--patience_epochs', type=int, default=5)

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--experiment_name', type=str, default='experiment')

    # Generation hyperparameters
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=512)

    # Evaluation frequency
    parser.add_argument('--eval_generate_every', type=int, default=3,
                        help="Run full generation-based eval every N epochs. "
                             "In between, only loss is computed (much faster).")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    args = parser.parse_args()
    return args


def generate_sql(args, model, encoder_input, encoder_mask, initial_decoder_input):
    """Beam search / greedy generation for one batch. Returns list of SQL strings."""
    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=initial_decoder_input,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            early_stopping=True,
        )
    return TOKENIZER.batch_decode(outputs, skip_special_tokens=True)


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    best_dev_loss = float('inf')
    epochs_since_improvement = 0

    model_type        = 'ft' if args.finetune else 'scr'
    experiment_name   = args.experiment_name
    checkpoint_dir    = os.path.join('checkpoints', f'{model_type}_experiments', experiment_name)
    gt_sql_path       = os.path.join('data/dev.sql')
    gt_record_path    = os.path.join('records/dev_gt_records.pkl')
    model_sql_path    = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')

    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)

    for epoch in range(args.max_n_epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        t0 = time.time()
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        train_time = time.time() - t0

        # ── Decide whether to run generation this epoch ───────────────────────
        last_epoch  = (epoch == args.max_n_epochs - 1)
        do_generate = (epoch % args.eval_generate_every == args.eval_generate_every - 1) or last_epoch

        t0 = time.time()
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader,
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path,
            do_generate=do_generate,
        )
        eval_time = time.time() - t0

        # ── Logging ───────────────────────────────────────────────────────────
        if do_generate:
            print(f"Epoch {epoch} [train {train_time:.1f}s | eval {eval_time:.1f}s]: "
                  f"train_loss={tr_loss:.4f}, dev_loss={eval_loss:.4f}, "
                  f"F1={record_f1:.4f}, EM={record_em:.4f}, SQL_EM={sql_em:.4f}, "
                  f"err={error_rate*100:.1f}%")
        else:
            print(f"Epoch {epoch} [train {train_time:.1f}s | eval {eval_time:.1f}s]: "
                  f"train_loss={tr_loss:.4f}, dev_loss={eval_loss:.4f} (loss only)")

        if args.use_wandb:
            log_dict = {
                'train/loss':     tr_loss,
                'dev/loss':       eval_loss,
                'train_time_sec': train_time,
                'eval_time_sec':  eval_time,
            }
            if do_generate:
                log_dict.update({
                    'dev/record_f1':  record_f1,
                    'dev/record_em':  record_em,
                    'dev/sql_em':     sql_em,
                    'dev/error_rate': error_rate,
                })
            wandb.log(log_dict, step=epoch)

        # ── Checkpointing & early stopping ────────────────────────────────────
        save_model(checkpoint_dir, model, best=False)

        if do_generate:
            if record_f1 > best_f1:
                best_f1 = record_f1
                epochs_since_improvement = 0
                save_model(checkpoint_dir, model, best=True)
                print(f"  ✓ New best F1: {best_f1:.4f} — checkpoint saved.")
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= args.patience_epochs:
                print(f"Early stopping after {epoch+1} epochs "
                      f"({args.patience_epochs} eval cycles without F1 improvement).")
                break
        else:
            # Track dev loss between generation evals (no early stopping trigger)
            if eval_loss < best_dev_loss:
                best_dev_loss = eval_loss


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss   = 0
    total_tokens = 0
    criterion    = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader, desc="Train"):
        optimizer.zero_grad()
        encoder_input   = encoder_input.to(DEVICE)
        encoder_mask    = encoder_mask.to(DEVICE)
        decoder_input   = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss    = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens    = torch.sum(non_pad).item()
            total_loss   += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens


def eval_epoch(args, model, dev_loader, gt_sql_path, model_sql_path,
               gt_record_path, model_record_path, do_generate=True):
    """
    do_generate=False : only compute cross-entropy loss (fast, GPU stays busy)
    do_generate=True  : also run beam search + compute F1/EM metrics (slow)
    """
    model.eval()
    criterion    = nn.CrossEntropyLoss()
    total_loss   = 0
    total_tokens = 0
    all_predictions = []

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_dec_input \
                in tqdm(dev_loader, desc="Eval"):

            encoder_input     = encoder_input.to(DEVICE)
            encoder_mask      = encoder_mask.to(DEVICE)
            decoder_input     = decoder_input.to(DEVICE)
            decoder_targets   = decoder_targets.to(DEVICE)
            initial_dec_input = initial_dec_input.to(DEVICE)

            # Loss (always computed)
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']
            non_pad       = decoder_targets != PAD_IDX
            loss          = criterion(logits[non_pad], decoder_targets[non_pad])
            num_tokens    = torch.sum(non_pad).item()
            total_loss   += loss.item() * num_tokens
            total_tokens += num_tokens

            # Generation (only when requested)
            if do_generate:
                batch_preds = generate_sql(args, model, encoder_input,
                                           encoder_mask, initial_dec_input)
                all_predictions.extend(batch_preds)

    eval_loss = total_loss / total_tokens

    if not do_generate:
        return eval_loss, -1.0, -1.0, -1.0, -1.0

    save_queries_and_records(all_predictions, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_path=gt_sql_path,
        model_path=model_sql_path,
        gt_query_records=gt_record_path,
        model_query_records=model_record_path,
    )
    error_rate = sum(1 for msg in error_msgs if msg != "") / len(error_msgs)
    return eval_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for encoder_input, encoder_mask, initial_dec_input in tqdm(test_loader, desc="Test"):
            encoder_input     = encoder_input.to(DEVICE)
            encoder_mask      = encoder_mask.to(DEVICE)
            initial_dec_input = initial_dec_input.to(DEVICE)

            batch_preds = generate_sql(args, model, encoder_input,
                                       encoder_mask, initial_dec_input)
            all_predictions.extend(batch_preds)

    save_queries_and_records(all_predictions, model_sql_path, model_record_path)
    print(f"Test inference done — {len(all_predictions)} predictions → {model_sql_path}")


def main():
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    
    gt_sql_path    = os.path.join('data/dev.sql')
    gt_record_path = os.path.join('records/dev_gt_records.pkl')

    if not os.path.exists(gt_record_path):
        print(f"Pre-computing ground-truth records → {gt_record_path}")
        gt_qs = read_queries(gt_sql_path)
        save_queries_and_records(gt_qs, gt_sql_path, gt_record_path)
        print("Done.")

    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Final evaluation with best checkpoint
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    experiment_name   = args.experiment_name
    model_type        = 'ft' if args.finetune else 'scr'
    gt_sql_path       = os.path.join('data/dev.sql')
    gt_record_path    = os.path.join('records/dev_gt_records.pkl')
    model_sql_path    = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')

    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader,
        gt_sql_path, model_sql_path,
        gt_record_path, model_record_path,
        do_generate=True,
    )
    print(f"\n=== Final Dev Results ===")
    print(f"Loss={dev_loss:.4f}, F1={dev_record_f1:.4f}, "
          f"EM={dev_record_em:.4f}, SQL_EM={dev_sql_em:.4f}, "
          f"err={dev_error_rate*100:.1f}%")

    model_sql_path    = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)


if __name__ == "__main__":
    main()
