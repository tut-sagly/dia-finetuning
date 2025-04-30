import argparse
import logging
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler
import torch.nn.functional as F
import bitsandbytes as bnb
from tqdm import tqdm
from datasets import load_dataset, interleave_datasets
from huggingface_hub import hf_hub_download
import math
import gc

import dac
from .config import DiaConfig
from .layers import DiaModel
from .model import Dia
from .audio import build_delay_indices, apply_audio_delay
from.dataset import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

LANG2BYTE = {
    "en": 3,
    "de": 4,
    "fr": 5,
    "es": 6,
    "it": 7,
    "nl": 14,
    "pl": 15,
    "pt": 16,
    "tr": 17,
    "hu": 18,
    
}

@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 2
    grad_accum_steps: int = 2
    learning_rate: float = 1e-5
    warmup_percentage: float = 0.01
    unconditional_frac: float = 0.15
    eval_step: int = 200
    save_step: int = 2000
    split_ratio: float = 0.9997
    shuffle_buffer_size: int = None  # for streaming shuffle
    seed: int = 42                # seed for reproducibility
    runs_dir: Path = Path("runs")
    run_name: str = "dia_finetune_langtag"
    output_dir: Path = Path(".")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Dia audio model")
    parser.add_argument("--config",    type=Path, default=Path("dia/config.json"))
    parser.add_argument("--dataset",   type=str,  default="Paradoxia/opendata-iisys-hui",
                        help="HuggingFace dataset name (if not using --csv_path).")
    parser.add_argument("--dataset2",  type=str,  default=None,
                        help="(Optional) second HF dataset to interleave (streaming)")
    parser.add_argument("--hub_model", type=str,  default="nari-labs/Dia-1.6B")
    parser.add_argument("--local_ckpt", type=str,  default=None)
    parser.add_argument("--csv_path",  type=Path, default=None,
                        help="Path to local CSV/TSV file with `audio|text` (if you want to train locally).")
    parser.add_argument("--audio_root",type=Path, default=None,
                        help="Root directory for local audio files (required if --csv_path is set).")
    parser.add_argument("--run_name",  type=str,  default=None)
    parser.add_argument("--output_dir",type=Path, default=None)
    parser.add_argument("--shuffle_buffer_size", type=int, default=None,
                        help="Buffer size for streaming dataset shuffle.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


'''class LocalDiaDataset(Dataset):
    """Load from a local CSV (sep='|') + an audio folder."""
    def __init__(self, csv_path: Path, audio_root: Path, config: DiaConfig, dac_model: dac.DAC):
        self.df = pd.read_csv(csv_path, sep=r"\s*\|\s*", engine="python",
                              names=["audio","text"] )
        self.audio_root = audio_root
        self.config = config
        self.dac_model = dac_model

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = row["text"]
        audio_path = self.audio_root / row["audio"]
        waveform, sr = torchaudio.load(audio_path)
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        waveform = waveform.unsqueeze(0)
        with torch.no_grad():
            audio_tensor = self.dac_model.preprocess(
                waveform, 44100
            ).to(next(self.dac_model.parameters()).device)
            _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
            encoded = encoded.squeeze(0).transpose(0, 1)
        return text, encoded, waveform



class HFDiaDataset(Dataset):
    """Wrap a HuggingFace `datasets.Dataset` (streaming or not) that has `audio.array` & `text`."""
    def __init__(self, hf_dataset, config: DiaConfig, dac_model: dac.DAC):
        self.dataset = hf_dataset
        self.config = config
        self.dac_model = dac_model

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        lang = sample.get("language", "")
        text = f"[{lang}]" + sample["text"]
        audio_info = sample["audio"]
        waveform = torch.tensor(audio_info["array"], dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        sr = audio_info.get("sampling_rate", 44100)
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        with torch.no_grad():
            audio_tensor = (
                self.dac_model.preprocess(waveform, 44100)
                .to(next(self.dac_model.parameters()).device)
            )
            _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
            encoded = encoded.squeeze(0).transpose(0, 1)
        return text, encoded, waveform



# ---------- streaming iterable wrapper ----------
class HFDiaIterDataset(torch.utils.data.IterableDataset):
    """Iterable wrapper for a HF streaming Dataset that has `audio.array` & `text`."""
    def __init__(self, hf_iterable, config: DiaConfig, dac_model: dac.DAC):
        super().__init__()
        self.iterable = hf_iterable
        self.config = config
        self.dac_model = dac_model

    def __iter__(self):
        for sample in self.iterable:
            lang = sample.get("language", "")
            text = f"[{lang}]" + sample.get('text_scribe')
            audio_info = sample['audio']
            waveform = torch.tensor(audio_info['array'], dtype=torch.float32)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            sr = audio_info.get('sampling_rate', 44100)
            if sr != 44100:
                waveform = torchaudio.functional.resample(waveform, sr, 44100)
            with torch.no_grad():
                audio_tensor = (
                    self.dac_model.preprocess(waveform, 44100)
                    .to(next(self.dac_model.parameters()).device)
                )
                _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
                encoded = encoded.squeeze(0).transpose(0, 1)
            yield text, encoded, waveform'''



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Enable CuDNN autotuner





def collate_fn(batch, config: DiaConfig, device: torch.device):
    from torch.nn.functional import pad

    texts, encodings, waveforms = zip(*batch)

    # -- Text inputs ---------------------------------------------------------
    max_text = config.data.text_length
    pad_tok = config.data.text_pad_value
    text_ids = []
    for txt in texts:
        b_full = txt.encode('utf-8')
        # replace leading "[lang]" prefix
        for code, val in LANG2BYTE.items():
            prefix = f"[{code}]".encode('utf-8')
            if b_full.startswith(prefix):
                b_full = bytes([val]) + b_full[len(prefix):]
                break
        bts = b_full[:max_text]
        arr = list(bts) + [pad_tok] * (max_text - len(bts))
        text_ids.append(torch.tensor(arr, dtype=torch.long))
    src = torch.stack(text_ids).to(device)
    src_pos = torch.arange(max_text, device=device).unsqueeze(0).expand(src.size(0), -1)
    src_pad = src.ne(pad_tok)
    enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    # -- Audio codes --------------------------------------------------------
    max_audio = config.data.audio_length
    # per-sample lengths (clipped to max_audio)
    seq_lens = [min(e.size(0), max_audio) for e in encodings]
    batch_max = max(seq_lens)

    # pad or trim each encoding to the batch max length
    padded = [pad(e, (0, 0, 0, batch_max - e.size(0))) if e.size(0) < batch_max else e[:batch_max]
              for e in encodings]
    codes = torch.stack(padded).to(device)  # (B, T=batch_max, C)

    B, T, C = codes.shape
    t_idx, idxs = build_delay_indices(B, T, C, config.data.delay_pattern)
    delayed = apply_audio_delay(
        codes,
        config.data.audio_pad_value,
        config.data.audio_bos_value,
        (t_idx, idxs)
    )
    # ensure no longer than max_audio
    delayed = delayed[:, :max_audio, :]

    # -- Targets with per-sample EOS ----------------------------------------

    max_tgt_len = max_audio + 2
    pad_val = config.data.audio_pad_value
    bos_val = config.data.audio_bos_value
    eos_val = config.data.audio_eos_value

    tgt = torch.full((B, max_tgt_len, C), pad_val, dtype=torch.long, device=device)
    tgt[:, 0, :] = bos_val
    tgt_lens = []
    for i, L in enumerate(seq_lens):
        tgt[i, 1:1 + L, :] = delayed[i, :L, :]
        tgt[i, 1 + L, :] = eos_val
        tgt_lens.append(1 + L + 1)

    tgt_pos = torch.arange(max_tgt_len, device=device).unsqueeze(0).expand(B, -1)
    tgt_pad = tgt.ne(pad_val).any(-1)

    causal = torch.tril(torch.ones((max_tgt_len, max_tgt_len),
                                    dtype=torch.bool,
                                    device=device))
    dec_self_attn_mask = (tgt_pad.unsqueeze(2) & tgt_pad.unsqueeze(1) & causal).unsqueeze(1)
    dec_cross_attn_mask = (tgt_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    return {
        'src_tokens': src,
        'src_positions': src_pos,
        'enc_self_attn_mask': enc_self_attn_mask,
        'tgt_tokens': tgt,
        'tgt_positions': tgt_pos,
        'dec_self_attn_mask': dec_self_attn_mask,
        'dec_cross_attn_mask': dec_cross_attn_mask,
        'waveforms': waveforms,
        'raw_text': texts[0],
        # per-sample lengths including BOS and EOS
        'tgt_lens': torch.tensor(tgt_lens, dtype=torch.long, device=device),
    }

def setup_loaders(dataset, dia_cfg: DiaConfig, train_cfg: TrainConfig, device):
    collate = lambda b: collate_fn(b, dia_cfg, device)
    if isinstance(dataset, HFDiaDataset):
        total = dataset.total_examples if dataset.total_examples is not None else dataset.dataset.info.splits['train'].num_examples
        n_train = int(train_cfg.split_ratio * total)
        n_val = total - n_train
        if n_val <= 0:
            raise RuntimeError(f"No validation samples: total={total}, split_ratio={train_cfg.split_ratio}")
        base = dataset.dataset.shuffle(buffer_size=train_cfg.shuffle_buffer_size, seed=train_cfg.seed) if train_cfg.shuffle_buffer_size else dataset.dataset
        val_stream = base.take(n_val)
        train_stream = base.skip(n_val)
        train_ds = HFDiaIterDataset(train_stream, dia_cfg, dataset.dac_model)
        val_ds = HFDiaIterDataset(val_stream, dia_cfg, dataset.dac_model)
        train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate)
        train_loader.steps_per_epoch = math.ceil(n_train / train_cfg.batch_size)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)
        return train_loader, val_loader
    ds_len = len(dataset)
    n_train = int(train_cfg.split_ratio * ds_len)
    train_ds, val_ds = random_split(dataset, [n_train, ds_len - n_train])
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)
    return train_loader, val_loader



def setup_optimizer_and_scheduler(model, train_loader, train_cfg):
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=train_cfg.learning_rate)
    # Determine steps per epoch: prefer len(), else use attached attribute
    try:
        steps_per_epoch = len(train_loader)
    except TypeError:
        if hasattr(train_loader, 'steps_per_epoch'):
            steps_per_epoch = train_loader.steps_per_epoch
        else:
            raise RuntimeError("Cannot determine steps_per_epoch for streaming loader")
    total_training_steps = steps_per_epoch * train_cfg.epochs
    sched = get_scheduler(
        'cosine', opt,
        num_warmup_steps=int(train_cfg.warmup_percentage * total_training_steps),
        num_training_steps=total_training_steps
    )
    return opt, sched



def train_step(model, batch, dia_cfg, train_cfg, opt, sched, writer, step, global_step):
    """
    Perform a single training step: forward, loss, backward, update, log.
    Now uses per‑sample tgt_lens to mask out padding after each EOS,
    and applies 4× loss weight on the first channel.
    """
    # (optional) unconditional conditioning
    if random.random() < train_cfg.unconditional_frac:
        pad_tok = dia_cfg.data.text_pad_value
        batch['src_tokens'] = torch.zeros_like(batch['src_tokens'])
        batch['enc_self_attn_mask'] = torch.zeros_like(batch['enc_self_attn_mask'])
        batch['dec_cross_attn_mask'] = torch.zeros_like(batch['dec_cross_attn_mask'])

    with autocast():
        # forward pass
        logits = model(
            src_BxS=batch['src_tokens'],
            tgt_BxTxC=batch['tgt_tokens'],
            src_positions=batch['src_positions'],
            tgt_positions=batch['tgt_positions'],
            enc_self_attn_mask=batch['enc_self_attn_mask'],
            dec_self_attn_mask=batch['dec_self_attn_mask'],
            dec_cross_attn_mask=batch['dec_cross_attn_mask'],
            enable_dropout=True,
        )
        # fetch per-sample target‑lengths (including BOS+frames+EOS)
        lens = batch['tgt_lens']                   # shape: (B,)
        max_L = int(lens.max().item())             # maximum over batch

        # keep only up through the last possible EOS slot
        # logits: (B, T, C, V) -> (B, max_L-1, C, V)
        logits = logits[:, : max_L - 1]

        # targets: shift off the BOS so 0..<max_L-1> align with logits
        # target: (B, T, C) -> (B, max_L-1, C)
        target = batch['tgt_tokens'][:, 1:max_L, :]

        B, Tm1, C = target.shape
        pad_val = dia_cfg.data.audio_pad_value

        # build a mask [B x (max_L-1)] that is True for t < (lens[i]-1)
        time_idx = torch.arange(Tm1, device=lens.device).unsqueeze(0)  # (1, Tm1)
        valid_time = time_idx < (lens.unsqueeze(1) - 1)                # (B, Tm1)
        mask = valid_time.unsqueeze(-1).expand(-1, -1, C)             # (B, Tm1, C)

        # apply 4× weight on first channel, 1× on others
        channel_weights = [4.0] + [1.0] * (C - 1)
        loss_c = 0.0
        _, _, _, V = logits.size()

        for c, w in enumerate(channel_weights):
            # flatten this channel
            lc = logits[:, :, c, :].reshape(-1, V)   # (B*Tm1, V)
            tc = target[:, :, c].reshape(-1)         # (B*Tm1,)
            mc = mask[:, :, c].reshape(-1)           # (B*Tm1,)

            # mask out padding and compute cross-entropy
            lc_valid = lc[mc]
            tc_valid = tc[mc]
            loss_c += w * F.cross_entropy(
                lc_valid, tc_valid,
                ignore_index=pad_val
            )

        # normalize by sum of weights
        loss = loss_c / sum(channel_weights)

    # scale + backward
    loss = loss / train_cfg.grad_accum_steps
    loss.backward()

    # step & log
    if (step + 1) % train_cfg.grad_accum_steps == 0:
        opt.step()
        sched.step()
        opt.zero_grad()
        true_loss = loss.item() * train_cfg.grad_accum_steps
        writer.add_scalar('Loss/train', true_loss, global_step)

    return loss.item() * train_cfg.grad_accum_steps



def eval_step(model, val_loader, dia_cfg, dac_model, writer, global_step):
    """
    Run evaluation: compute average loss on validation set and log audio samples.
    """
    eval_losses = []
    last_batch = None
    with torch.inference_mode():
        for eb in tqdm(val_loader, desc="eval"):
            last_batch = eb

            # 1) do your forward in mixed precision
            with autocast():
                logits16 = model(
                    src_BxS=eb['src_tokens'],
                    tgt_BxTxC=eb['tgt_tokens'],
                    src_positions=eb['src_positions'],
                    tgt_positions=eb['tgt_positions'],
                    enc_self_attn_mask=eb['enc_self_attn_mask'],
                    dec_self_attn_mask=eb['dec_self_attn_mask'],
                    dec_cross_attn_mask=eb['dec_cross_attn_mask'],
                    enable_dropout=False,
                )[:, :-1]

            logits = logits16.float()
            target = eb['tgt_tokens'][:, 1:]
            B_e, T_e, C_e = target.shape
            V_e = logits.size(-1)

            loss_e = 0.0
            weights_e = [4.0] + [1.0] * (C_e - 1)
            for c, w in enumerate(weights_e):
                lc = logits[:, :, c, :].reshape(-1, V_e)
                tc = target[:, :, c].reshape(-1)
                loss_e += w * F.cross_entropy(
                    lc, tc, ignore_index=dia_cfg.data.audio_pad_value
                )
            loss_e = loss_e / sum(weights_e)

            eval_losses.append(loss_e)

    avg_eval_loss = sum(eval_losses) / len(eval_losses)
    writer.add_scalar('Loss/eval', avg_eval_loss.item(), global_step)

    try:
        orig_dtype = next(model.parameters()).dtype
        model = model.float()
        dia_gen = Dia(dia_cfg, device)
        dia_gen.model, dia_gen.dac_model = model, dac_model
        with torch.inference_mode():
            audio_de = dia_gen.generate(text="[de]Das neue Samsung Galaxy S 21 Ultra 5G. Nur bis zum einundreißigsten Januar. Jetzt bei Media Markt.")#last_batch['raw_text'])
            audio_en = dia_gen.generate(text="[en]Whether you want to train your own model from scratch or adapt a pre-trained model for your own use case, generally the larger part of the engineering effort goes into pre-processing the dataset for training.")
        writer.add_audio('Eval/de', audio_de, global_step, 44100)
        writer.add_audio('Eval/en', audio_en, global_step, 44100)
        del audio_de, audio_en   
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception:
        logger.exception("Eval error")
    
    finally:
        if orig_dtype == torch.float16:
            model = model.half()


def train(model, dia_cfg: DiaConfig, dac_model: dac.DAC, dataset, train_cfg: TrainConfig):
    """
    Run the full training loop over epochs.
    """
    # prepare directories
    train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (train_cfg.runs_dir / train_cfg.run_name).mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    train_loader, val_loader = setup_loaders(dataset, dia_cfg, train_cfg, device)
    opt, sched = setup_optimizer_and_scheduler(model, train_loader, train_cfg)

    writer = SummaryWriter(train_cfg.runs_dir / train_cfg.run_name)
    model.train()

    steps_per_epoch = getattr(train_loader, 'steps_per_epoch', None)
    if steps_per_epoch is None:
        try:
            steps_per_epoch = len(train_loader)
        except Exception:
            steps_per_epoch = None

    for epoch in range(train_cfg.epochs):
        # iterate with progress bar, using total if known
        loader_iter = tqdm(
            train_loader,
            desc=f"E{epoch+1}",
            total=steps_per_epoch
        )
        for step, batch in enumerate(loader_iter):
            global_step = epoch * (steps_per_epoch or 0) + step
            # training step
            loss=train_step(model, batch, dia_cfg, train_cfg, opt, sched, writer,step, global_step)

            cur_alloc = torch.cuda.memory_allocated()   # bytes currently allocated by tensors
            peak_alloc = torch.cuda.max_memory_allocated()  # bytes peak during program
            # optionally convert to GB
            cur_gb  = cur_alloc  / 1024**3
            peak_gb = peak_alloc / 1024**3

            # update the tqdm postfix
            loader_iter.set_postfix({
                'loss': f"{loss:.4f}",
                'VRAM (GB)': f"{cur_gb:.2f}/{peak_gb:.2f}"
            })

            # remember to zero the peak if you want rolling peaks per step
            torch.cuda.reset_peak_memory_stats()
            

            # evaluation
            if step % train_cfg.eval_step == 0:
                model.eval()
                with torch.no_grad():
                    eval_step(model, val_loader, dia_cfg, dac_model, writer, global_step)
                model.train()

            # checkpoint
            if step and step % train_cfg.save_step == 0:
                ckpt = train_cfg.output_dir / f"ckpt_step{global_step}.pth"
                torch.save(model.state_dict(), ckpt)
                logger.info(f"Saved checkpoint: {ckpt}")

        # end of epoch checkpoint
        ckpt_e = train_cfg.output_dir / f"ckpt_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_e)
        logger.info(f"Saved end-of-epoch checkpoint: {ckpt_e}")


def main():
    args = get_args()
    dia_cfg = DiaConfig.load(args.config)
    dac_model = dac.DAC.load(dac.utils.download()).to(device)

    # choose dataset
    if args.csv_path:
        if not args.audio_root:
            raise ValueError("`--audio_root` must be set when using `--csv_path`")
        dataset = LocalDiaDataset(args.csv_path, args.audio_root, dia_cfg, dac_model)
    else:
        # load one or two streaming HF datasets
        ds1 = load_dataset(args.dataset, split="train", streaming=True)
        if args.dataset2:
            ds2 = load_dataset(args.dataset2, split="train", streaming=True)
            # sum their lengths
            total1 = ds1.info.splits['train'].num_examples
            total2 = ds2.info.splits['train'].num_examples
            total = total1 + total2
            hf_ds = interleave_datasets([ds1, ds2])
            dataset = HFDiaDataset(hf_ds, dia_cfg, dac_model)
            # attach total examples for loader
            dataset.total_examples = total
        else:
            hf_ds = ds1
            dataset = HFDiaDataset(hf_ds, dia_cfg, dac_model)

    train_cfg = TrainConfig(
        run_name   = args.run_name   or TrainConfig.run_name,
        output_dir = args.output_dir or TrainConfig.output_dir,
        shuffle_buffer_size = args.shuffle_buffer_size,
        seed = args.seed,
    )

    # load model checkpoint
    if args.local_ckpt:
        ckpt_file = args.local_ckpt
    else:
        ckpt_file = hf_hub_download(args.hub_model, filename="dia-v0_1.pth")
    model = DiaModel(dia_cfg)
    model.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
    model=model.half()
    model = torch.compile(model, backend="inductor")

    # start training
    train(model, dia_cfg, dac_model, dataset, train_cfg)


if __name__ == "__main__":
    main()

