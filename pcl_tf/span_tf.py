import torch
import torch.nn as nn
from transformers import AutoModel
from types import SimpleNamespace
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    Trainer
)

class SpanModel(nn.Module):
    """Encoder with two heads trained jointly:
    1) category_head: [CLS] → 7 PCL category logits (auxiliary signal)
    2) binary_head:   concat([CLS], category_logits) → binary PCL prediction
    The category predictions are a *feature* for the binary decision."""

    def __init__(self, checkpoint=None, num_categories=7, dropout=0.1, cache_dir=None,
                 encoder=None):
        super().__init__()
        # Accept pre-built encoder (cached) or load from checkpoint
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = AutoModel.from_pretrained(checkpoint, cache_dir=cache_dir)
        h = self.encoder.config.hidden_size  # 1024 for ALBERT-large

        self.dropout = nn.Dropout(dropout)

        # Auxiliary: predict which PCL categories are present
        self.category_head = nn.Linear(h, num_categories)

        # Main: binary PCL detection, informed by category logits
        self.binary_head = nn.Sequential(
            nn.Linear(h + num_categories, h // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h // 2, 2),
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = self.dropout(outputs.last_hidden_state[:, 0])  # [CLS] pooling

        cat_logits = self.category_head(cls)                       # (B, 7)
        binary_input = torch.cat([cls, cat_logits], dim=-1)        # (B, 1024+7)
        binary_logits = self.binary_head(binary_input)             # (B, 2)

        return binary_logits, cat_logits


class MultiTaskTrainer(Trainer):
    """Joint training: focal CE on binary + BCE on categories.
    Uses WeightedRandomSampler to balance the ~10:1 class imbalance."""

    def __init__(self, focal_alpha, focal_gamma, aux_weight,
                 weighted_sampler=None, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        # Store focal params directly — inline CE avoids one_hot + sigmoid + BCE overhead
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cat_loss_fn = nn.BCEWithLogitsLoss()
        self.aux_weight = aux_weight
        self.weighted_sampler = weighted_sampler
        # Per-class weights for cross-entropy (handles imbalance in loss, not data)
        self.class_weights = class_weights  # torch.Tensor of shape (num_classes,) or None

    def _focal_ce(self, logits, labels):
        """Focal cross-entropy with optional per-class weights.

        Key: pt (model confidence) must be computed from RAW logits, not from
        weighted CE. Otherwise class_weights corrupt the focal modulation and
        inflate the training loss to nonsensical values (6+ at epoch 1)."""
        # pt = P(correct class) — from unweighted softmax
        pt = torch.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)

        # Standard CE with optional per-class weights
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce = F.cross_entropy(logits, labels, weight=w, reduction="none")  # (B,)

        # Focal modulation based on true confidence
        focal_weight = self.focal_alpha * (1 - pt.detach()) ** self.focal_gamma
        return (focal_weight * ce).mean()

    def get_train_dataloader(self):
        """Override to inject WeightedRandomSampler for class balancing."""
        if self.weighted_sampler is None:
            return super().get_train_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self.weighted_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        cat_labels = inputs.pop("cat_labels")

        binary_logits, cat_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
        )

        loss_binary = self._focal_ce(binary_logits, labels)
        loss_cat = self.cat_loss_fn(cat_logits, cat_labels)
        loss = loss_binary + self.aux_weight * loss_cat

        if return_outputs:
            return loss, SimpleNamespace(logits=binary_logits)
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to handle multi-task forward during evaluation."""
        with torch.no_grad():
            labels = inputs.pop("labels").to(self.args.device)
            cat_labels = inputs.pop("cat_labels").to(self.args.device)
            inputs = self._prepare_inputs(inputs)

            binary_logits, cat_logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
            )

            loss_binary = self._focal_ce(binary_logits, labels)
            loss_cat = self.cat_loss_fn(cat_logits, cat_labels)
            loss = loss_binary + self.aux_weight * loss_cat

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, binary_logits, labels)

