from transformers import (
    GenerationMixin, EncoderDecoderCache,
)
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
)
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup
import os
import json
from pathlib import Path
from typing import Dict, List
import re
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer,
)

from CopiedFromYam.ASR.CONDFORMER_BLOCKS import CondFormerBlock
from CopiedFromYam.ASR.utils import group_files_by_name, remove_unwanted_text_path
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from jiwer import wer
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

from transformers import WhisperForConditionalGeneration, WhisperConfig
from transformers.models.whisper.modeling_whisper import (
    BaseModelOutput,
)
import torch
import torch.nn as nn
from typing import Optional, Tuple
# from peft import get_peft_model, LoraConfig, TaskType

class CustomWhisperEncoder(nn.Module):
    def __init__(self, config: WhisperConfig, method: str = "film_basic"):
        super().__init__()
        from transformers.models.whisper.modeling_whisper import WhisperEncoder
        self.method = method
        self.inner  = WhisperEncoder(config)
        self.d_model = config.d_model
        self.config=config
        self.conv1 = self.inner.conv1
        self.conv2 = self.inner.conv2
        self.layers=self.inner.layers
        self.noise_conv1 = nn.Conv1d(self.inner.num_mel_bins,   self.d_model, kernel_size=3, padding=1)
        self.noise_conv2 = nn.Conv1d(  self.d_model,   self.d_model, kernel_size=3, stride=2, padding=1)

        nn.init.kaiming_normal_(self.noise_conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.noise_conv2.weight, nonlinearity="relu")

        if self.method == "film_basic":
            self.noise_to_gamma = nn.ModuleList([nn.Conv1d(self.d_model, self.d_model, 1)])
            self.noise_to_beta = nn.ModuleList([nn.Conv1d(self.d_model, self.d_model, 1)])
            for gamma, beta in zip(self.noise_to_gamma, self.noise_to_beta):
                nn.init.zeros_(gamma.weight)
                nn.init.zeros_(beta.weight)
        elif self.method=='multiple_film':
            self.num_layers = config.num_hidden_layers  # should be 12 for whisper-small
            self.noise_to_gamma = nn.ModuleList([
                nn.Conv1d(self.d_model, self.d_model, 1) for _ in range(self.num_layers)
            ])
            self.noise_to_beta = nn.ModuleList([
                nn.Conv1d(self.d_model, self.d_model, 1) for _ in range(self.num_layers)
            ])
            for gamma, beta in zip(self.noise_to_gamma, self.noise_to_beta):
                nn.init.zeros_(gamma.weight)
                nn.init.zeros_(beta.weight)
        elif self.method=='condformer':
            self.condblock= nn.Sequential(*[
                CondFormerBlock(dim=int(dim_conformer * 2 ** 3), z_dim=dim_conformer, num_heads=heads_conformer,
                                ffn_expansion_factor=ffn_expansion_factor,
                                bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks_conformer)])

    def run_film_block(self,hs,idx,noise_embeds):
        gamma = self.noise_to_gamma[idx](noise_embeds.permute(0, 2, 1)).transpose(1, 2)
        beta = self.noise_to_beta[idx](noise_embeds.permute(0, 2, 1)).transpose(1, 2)
        hidden_states = hs * (1 + gamma) + beta
        return hidden_states

    def forward(
        self,
        input_features: torch.FloatTensor,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> BaseModelOutput:

        r"""
              Args:
                  input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                      Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                      obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                      `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                      `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                      and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
                  attention_mask (`torch.Tensor`)`, *optional*):
                      Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                      but it is not used. By default the silence in the input log mel spectrogram are ignored.
                  head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                      Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                      - 1 indicates the head is **not masked**,
                      - 0 indicates the head is **masked**.
                  output_attentions (`bool`, *optional*):
                      Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                      returned tensors for more detail.
                  output_hidden_states (`bool`, *optional*):
                      Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                      for more detail.
                  return_dict (`bool`, *optional*):
                      Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
              """
        noise_features = input_features[:, 1].squeeze(1)
        input_features = input_features[:, 0].squeeze(1)
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        noise_embeds = nn.functional.gelu(self.noise_conv1(noise_features))
        noise_embeds = nn.functional.gelu(self.noise_conv2(noise_embeds)).permute(0, 2, 1)

        embed_pos = self.inner.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.inner.dropout, training=self.inner.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

## loop over the attentions blocks
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.method=='multiple_film':
                hidden_states=self.run_film_block(hidden_states,idx,noise_embeds)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.inner.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.inner.gradient_checkpointing and self.inner.training:
                    layer_outputs = self.inner._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.inner.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if self.method == "film_basic" and noise_features is not None:
            hidden_states = self.run_film_block(hidden_states, 0, noise_embeds)
        elif self.method=='condformer':
            hidden_states = self.condblock({'x': hidden_states.unsqueeze(1), 'z': noise_embeds.unsqueeze(1)})['x']

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)


        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class NoiseWhisperForConditionalGeneration(
        WhisperForConditionalGeneration, GenerationMixin):

    def __init__(self, config: WhisperConfig, method="film_basic"):
        super().__init__(config)
        self.model.encoder = CustomWhisperEncoder(config, method)
        self.method = method

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
            input_features=input_features,
            )

        return super().forward(
            input_features  = None,              # already consumed
            encoder_outputs = encoder_outputs,
            past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values),
            **kwargs
        )

    @classmethod
    def from_pretrained(cls, *model_args, method="film_basic", **kwargs):
        base = WhisperForConditionalGeneration.from_pretrained(*model_args, **kwargs)
        model = cls(base.config, method=method)
        model.load_state_dict(base.state_dict(), strict=False)
        model.model.encoder.inner.load_state_dict(base.model.encoder.state_dict())
        missing, _ = model.load_state_dict(base.state_dict(), strict=False)
        print("Missing after transplant:", [k for k in missing if "encoder" not in k])

        return model


class MelNoiseDataset(Dataset):
    """Loads .pt tensors containing (80,T) mel and noise + text."""

    def __init__(self, manifest: str | Path, tokenizer: PreTrainedTokenizer):
        self.tok = tokenizer
        self.rows = [json.loads(l) for l in Path(manifest).read_text().splitlines()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        m1 = re.search(r"actual_tbins=(\d+)", r["reverb"])
        m2 = re.search(r"actual_tbins=(\d+)", r["clean"])
        if m1 and int(m1.group(1))==int(m2.group(1)):
            true_T = int(m1.group(1)) + 1   # if your index is zero-based
        else:
            raise ValueError

        reverb   = np.load(r["reverb"])[:,:true_T]     # (80,T)
        clean    = np.load(r["clean"])[:,:true_T]            # (80,T)
        noise    = torch.from_numpy(clean-reverb)# (80,T)
        ids      = self.tok(r["text"]).input_ids
        results  = {"reverb": torch.from_numpy(reverb), "noise": noise, "labels": torch.tensor(ids, dtype=torch.long)}
        return results

    @staticmethod
    def collate_fn(batch: List[Dict]):
        max_T = 3000
        mels   = torch.stack([F.pad(item["reverb"], (0, max_T - item["reverb"].shape[-1])) for item in batch])
        noises = torch.stack([F.pad(item["noise"], (0, max_T - item["noise"].shape[-1])) for item in batch])
        labels = [item["labels"] for item in batch]
        pad_id = tokenizer.pad_token_id
        input_ids = pad_sequence(labels, batch_first=True, padding_value=pad_id)
        return {
            "input_features": mels,
            "noise_embedding": noises,
            "labels": input_ids,
        }


def create_manifest(reverb_path, clean_path, text_paths):
    data=[]
    for reverb, clean, text_file in zip(reverb_path, clean_path,  text_paths):
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        yield{
            "reverb": reverb,
            "clean": clean,
            "text": text
        }
    return data


def get_lists_of_paths(output_path):
    all_rows = []
    folders = os.listdir(output_path)
    for folder in folders:
        output_folder_path  = os.path.join(output_path, folder)
        targets_folder_path = os.path.join(targets_path, folder)
        clean_folder_path   = os.path.join(clean_path, folder)
        reverb_folder_path  = os.path.join(reverb_path, folder)
        file_groups = group_files_by_name(output_folder_path, WANTED_CH=WANTED_CH)
        file_groups = group_files_by_name(clean_folder_path,file_groups, WANTED_CH=WANTED_CH)
        file_groups = group_files_by_name(reverb_folder_path,file_groups, WANTED_CH=WANTED_CH)
        text_d = group_files_by_name(targets_folder_path, WANTED_CH=WANTED_CH)
        text_d['text'] = remove_unwanted_text_path(text_d['text'], file_groups['pred'])
        for row in create_manifest(file_groups["reverb"],
                                   file_groups["clean"],
                                   text_d["text"]):
            all_rows.append(row)
    random.shuffle(all_rows)
    n = len(all_rows)
    n_train = n // 2
    n_val = (n - n_train) // 2
    n_test = n - n_train - n_val

    splits = {
        'test': all_rows[:n_test],
        'val': all_rows[n_test:n_test + n_val],
        'train': all_rows[n_test + n_val:]
    }

    for split_name, split_data in splits.items():
        filename = f"{split_name}_{manifest_name}"
        with open(filename, "w", encoding='utf-8') as f:
            for row in split_data:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.

    for batch in tqdm(dataloader, desc="Training"):
        feats  = batch["input_features"].to(device).float()
        noises = batch["noise_embedding"].to(device).float()
        labels = batch["labels"].to(device)

        out = model(
            input_features=torch.cat([feats.unsqueeze(1), noises.unsqueeze(1)], dim=1),
            labels=labels,
            output_hidden_states=False,
        )

        loss = out.loss
        loss.backward()
        optimizer.step()
        scheduler.step()       # <— step the LR scheduler
        optimizer.zero_grad()
        total_loss += loss.item()
    print(scheduler.get_lr())
    return total_loss / len(dataloader)

def _clean(text: str) -> str:
    """Lower-case, keep letters only (English)."""
    return re.sub(r'[^a-z ]', '', text.lower())

def evaluate(model, dataloader, tokenizer, device,model_type=None):
    model.eval()
    total_loss, preds, refs = 0.0, [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            feats  = batch["input_features"].to(device).float()
            noises = batch["noise_embedding"].to(device).float()
            labels = batch["labels"].to(device)
            if model_type!='vanilla':
                out = model(
                    input_features=torch.cat([feats.unsqueeze(1), noises.unsqueeze(1)], dim=1),
                    labels=labels,
                    output_hidden_states=False,
                )

            else:
                out = model(input_features=feats,
                            labels=labels)
            total_loss += out.loss.item()

            attention_mask = (feats.abs().sum(dim=-1) > 0).long()

            if model_type!='vanilla':

                gen_ids = model.generate(
                    input_features=torch.cat([feats.unsqueeze(1), noises.unsqueeze(1)], dim=1),
                    attention_mask=attention_mask,
                    num_beams=5,
                    early_stopping=True,
                    repetition_penalty=1.2,
                )

            else:
                gen_ids = model.generate(
                    input_features=feats,
                    attention_mask=attention_mask,
                    num_beams=5,
                    early_stopping=True,
                    repetition_penalty=1.2,
                )
                # decode
            pred_strs = processor.batch_decode(gen_ids, skip_special_tokens=True)
            ref_strs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            preds += [_clean(t) for t in pred_strs]
            refs += [_clean(t) for t in ref_strs]
    wer_score = wer(refs, preds)
    start_idx = torch.randint(0, len(preds) - 3, (1,)).item()
    print("\n".join(f"PREDS: {p}\nTARGS: {r}" for p, r in zip(preds[start_idx:start_idx+3], refs[start_idx:start_idx+3])))
    return total_loss / len(dataloader), wer_score, refs, preds

def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    if METHOD != 'condformer':
        for param in model.model.encoder.noise_to_gamma.parameters():
            param.requires_grad = True
        for param in model.model.encoder.noise_to_beta.parameters():
            param.requires_grad = True
    else:
        for param in model.model.encoder.condblock.parameters():
            param.requires_grad = True

def set_globals():
    manifest_path=os.path.join(os.getcwd(), F'train_{manifest_name}')
    if not  os.path.exists(manifest_path):
        get_lists_of_paths(output_path)
    tokenizer  = WhisperTokenizer.from_pretrained("openai/whisper-small")
    train_ds   = MelNoiseDataset(f"train_{manifest_name}", tokenizer)
    val_ds     = MelNoiseDataset(f"val_{manifest_name}",   tokenizer)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor  = WhisperProcessor.from_pretrained(model_name)

    model  = NoiseWhisperForConditionalGeneration.from_pretrained(
                 model_name, method=METHOD
             ).to(device)

    freeze_backbone(model)
    model.config.forced_decoder_ids = None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=MelNoiseDataset.collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=MelNoiseDataset.collate_fn)
    optimizer    = optim.AdamW(model.parameters(), lr=lr)
    total_steps  = n_epochs * len(train_loader)
    warmup_steps = 5
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )
    return tokenizer, device, model, processor, train_loader, val_loader, optimizer, warmup_steps, scheduler

def run():
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_wer, refs, preds = evaluate(model, val_loader, tokenizer, device)
        print({"train_loss": train_loss, "val_loss": val_loss, "Val WER": f"{val_wer:.3%}", "epoch": epoch + 1})

    val_loss, val_wer, refs, preds = evaluate(model, val_loader, tokenizer, device)
    print({ "val_loss": val_loss, "Val WER": f"{val_wer:.3%}"})
    model_type = 'vanilla'
    vanilla = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    val_loss_vanilla, wer_vanilla, _, _ = evaluate(vanilla, val_loader, tokenizer, device,model_type)
    print(f"Vanilla Whisper → WER: {wer_vanilla:.3%}")

if __name__ == "__main__":

    targets_path = r'/storage/tal/thesis/DataBase_BIUREV/transcription_matched/test'
    clean_path   = r'/storage/tal/thesis/DataBase_BIUREV/clean_melspec/test'
    reverb_path  = r'/storage/tal/thesis/DataBase_BIUREV/reverb_melspec/test'
    output_path  = r'/storage/tal/thesis/DataBase_BIUREV/dereverb_mel/calibrated_rcps/test'

    manifest_name = "all.jsonl"
    WANTED_CH     = "ch1"
    model_name    = "openai/whisper-small"
    batch_size=8
    lr=0.0001
    n_epochs=50
    stage='train'
    METHOD='film_basic' #film_basic' #multiple_film'
    if METHOD=='condformer':
        dim_conformer= 2
        heads_conformer = 8
        ffn_expansion_factor = 2.66
        bias = False
        LayerNorm_type = 'BiasFree'
        num_blocks_conformer=8

    tokenizer, device, model, processor, train_loader, val_loader, optimizer, warmup_steps, scheduler=set_globals()
    run()

