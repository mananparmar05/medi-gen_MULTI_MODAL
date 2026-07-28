"""
Main Training Loop Class for the Multimodal Medical Report Generation system.

Orchestrates:
    - Multi-phase training (Phase 1: frozen GPT-2 body, Phase 2: unfrozen with discriminative LR).
    - Gradient accumulation for CPU execution stability under memory constraints.
    - Differentiable curriculum NLI loss injection.
    - Validation loop computing standard BLEU metrics + Factual Consistency Score (FCS).
    - Early stopping and checkpointing.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.losses import ConsistencyWeightedLoss
from training.scheduler import CurriculumLambdaScheduler, get_optimizer_params
from evaluation.metrics import compute_nlg_metrics
from utils.helpers import save_checkpoint

logger = logging.getLogger(__name__)


class ReportGenerationTrainer:
    """
    Manages main model optimization, validation, curriculum schedules, and checkpoint saves.
    
    Args:
        model: MultimodalReportGenerator instance.
        train_loader: DataLoader for train set.
        val_loader: DataLoader for val set.
        tokenizer: ReportTokenizer.
        config: Hyperparameter DotDict config.
        device: Compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        config,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = ConsistencyWeightedLoss(config)
        
        # Scheduler for consistency weight lambda
        self.lambda_scheduler = CurriculumLambdaScheduler(
            lambda_max=config.training.lambda_max,
            ramp_start=config.training.lambda_ramp_start,
            ramp_end=config.training.lambda_ramp_end,
        )
        
        # Setup optimizer groups
        opt_groups = get_optimizer_params(
            model=self.model,
            base_lr=config.training.learning_rate,
            discriminative_factor=config.decoder.discriminative_lr_factor,
            weight_decay=config.training.weight_decay,
        )
        
        self.optimizer = torch.optim.AdamW(opt_groups)
        
        # Standard learning rate decay
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.max_epochs,
            eta_min=1e-6
        )
        
        # Tracker variables
        self.best_fcs = -1.0
        self.best_bleu = -1.0
        self.patience_counter = 0
        
        # Create directories
        Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.training.log_dir).mkdir(parents=True, exist_ok=True)

    def resume_from_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model and optimizer state from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            
        Returns:
            The next epoch to resume training from (0-indexed).
        """
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
        completed_epoch = ckpt["epoch"]  # 0-indexed epoch that was completed
        
        # Advance the LR scheduler to the correct position
        for _ in range(completed_epoch + 1):
            self.lr_scheduler.step()
        
        # Restore best metrics if available
        if "val_metrics" in ckpt and ckpt["val_metrics"]:
            val_metrics = ckpt["val_metrics"]
            if "val_fcs" in val_metrics:
                self.best_fcs = val_metrics["val_fcs"]
            if "bleu_4" in val_metrics:
                self.best_bleu = val_metrics["bleu_4"]
        
        resume_epoch = completed_epoch + 1
        logger.info(f"Checkpoint loaded. Resuming from epoch {resume_epoch + 1} (0-indexed: {resume_epoch}).")
        return resume_epoch

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run single epoch training with gradient accumulation."""
        self.model.train()
        # Set training phase (e.g. freeze/unfreeze GPT-2 blocks)
        self.model.set_training_phase(epoch)
        
        current_lambda = self.lambda_scheduler.get_lambda(epoch)
        
        epoch_loss = 0.0
        components_sum = {"loss_total": 0.0, "loss_generation": 0.0, "loss_alignment": 0.0, "loss_consistency": 0.0}
        
        self.optimizer.zero_grad()
        
        # ProgressBar
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        accumulation_counter = 0
        
        for batch_idx, batch in enumerate(progress):
            # Move inputs to device
            frontal_img = batch["frontal_img"].to(self.device)
            lateral_img = batch["lateral_img"].to(self.device)
            lateral_mask = batch["lateral_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            alignment_targets = batch["alignment_targets"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                frontal_img=frontal_img,
                lateral_img=lateral_img,
                lateral_mask=lateral_mask,
                labels=labels,
                input_ids=input_ids,
                target_ids=target_ids,
                attention_mask=attention_mask,
                alignment_targets=alignment_targets,
            )
            
            # Compute loss
            loss, loss_details = self.criterion(
                generator_outputs=outputs,
                target_ids=target_ids,
                labels=labels,
                tokenizer=self.tokenizer,
                nli_scorer=self.model.nli_scorer,
                current_lambda=current_lambda,
                device=self.device,
            )
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.training.gradient_accumulation_steps
            loss.backward()
            
            # Update trackers
            epoch_loss += loss.item() * self.config.training.gradient_accumulation_steps
            for k in components_sum:
                if k in loss_details:
                    components_sum[k] += loss_details[k]
            
            accumulation_counter += 1
            
            # Optimizer step
            if accumulation_counter >= self.config.training.gradient_accumulation_steps:
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.training.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulation_counter = 0
                
            # Update progress bar info
            progress.set_postfix({
                "loss": f"{loss_details['loss_total']:.4f}",
                "gen": f"{loss_details['loss_generation']:.4f}",
                "nli": f"{loss_details['loss_consistency']:.4f}",
                "λ": f"{current_lambda:.2f}"
            })
            
        # Complete remaining gradient step if exists
        if accumulation_counter > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        # LR step
        self.lr_scheduler.step()
        
        # Average metrics
        num_batches = len(self.train_loader)
        avg_metrics = {k: v / num_batches for k, v in components_sum.items()}
        avg_metrics["lr"] = self.optimizer.param_groups[-1]["lr"]
        
        return avg_metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation loop, generating text reports and calculating metrics."""
        self.model.eval()
        
        val_loss = 0.0
        all_hypotheses = []
        all_references = []
        
        # NLI factual scores tracker
        fcs_scores = []
        contra_rates = []
        
        # Validation subset and frequency controls
        val_subset_size = self.config.training.get("val_subset_size", None)
        val_metrics_every_n_epochs = self.config.training.get("val_metrics_every_n_epochs", 1)
        compute_heavy_metrics = (epoch + 1) % val_metrics_every_n_epochs == 0
        
        val_samples_processed = 0
        
        progress = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        with torch.no_grad():
            for batch in progress:
                # Break early if we exceed subset size
                if val_subset_size is not None and val_samples_processed >= val_subset_size:
                    break
                    
                frontal_img = batch["frontal_img"].to(self.device)
                lateral_img = batch["lateral_img"].to(self.device)
                lateral_mask = batch["lateral_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Standard forward to compute validation loss
                outputs = self.model(
                    frontal_img=frontal_img,
                    lateral_img=lateral_img,
                    lateral_mask=lateral_mask,
                    labels=labels,
                    input_ids=input_ids,
                    target_ids=target_ids,
                    attention_mask=attention_mask,
                )
                
                loss_val = outputs["generation_loss"]
                val_loss += loss_val.item()
                
                batch_size = labels.size(0)
                val_samples_processed += batch_size
                
                if compute_heavy_metrics:
                    # Autoregressive generation to compute BLEU & NLI scores
                    gen_outputs = self.model.generate(
                        frontal_img=frontal_img,
                        lateral_img=lateral_img,
                        lateral_mask=lateral_mask,
                        labels=labels,
                        tokenizer=self.tokenizer,
                        max_length=self.config.decoder.max_gen_length,
                    )
                    
                    generated_texts = gen_outputs["generated_texts"]
                    gt_texts = batch["report_text"]
                    
                    all_hypotheses.extend(generated_texts)
                    all_references.extend(gt_texts)
                    
                    # Factual consistency scoring (batched)
                    nli_results = self.model.nli_scorer.score_reports_batch(
                        finding_labels=labels,
                        generated_texts=generated_texts,
                        finding_label_names=self.config.data.finding_labels,
                        finding_templates=self.config.data.finding_templates,
                        device=self.device,
                    )
                    fcs_scores.extend(nli_results["fcs"])
                    contra_rates.extend(nli_results["contradiction_rate"])
                    
        # Calculate final average validation loss
        num_batches_processed = val_samples_processed / self.config.training.batch_size
        avg_val_loss = val_loss / max(num_batches_processed, 1.0)
        
        metrics = {
            "val_loss": avg_val_loss,
        }
        
        if compute_heavy_metrics and all_hypotheses:
            # Compute NLG metrics (BLEU, METEOR, ROUGE-L, CIDEr)
            nlg_metrics = compute_nlg_metrics(all_hypotheses, all_references)
            
            # Calculate final averages
            avg_fcs = np.mean(fcs_scores) if fcs_scores else 0.0
            avg_cr = np.mean(contra_rates) if contra_rates else 0.0
            
            metrics.update({
                "val_fcs": avg_fcs,
                "val_cr": avg_cr,
                **nlg_metrics
            })
            
        return metrics

    def train(self, start_epoch: int = 0) -> None:
        """Run complete multi-epoch training process.
        
        Args:
            start_epoch: Epoch to start from (0-indexed). Used when resuming from checkpoint.
        """
        max_epochs = self.config.training.max_epochs
        if start_epoch > 0:
            logger.info(f"Resuming training from epoch {start_epoch + 1}/{max_epochs}...")
        else:
            logger.info(f"Starting model optimization for {max_epochs} epochs...")
        
        for epoch in range(start_epoch, max_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Save a pre-validation checkpoint so weights are never lost if validation crashes
            pre_val_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_metrics": train_metrics,
                "config": self.config,
            }
            save_checkpoint(
                state=pre_val_state,
                checkpoint_dir=self.config.training.checkpoint_dir,
                filename="checkpoint_latest.pt",
                is_best=False,
            )
            logger.info(f"Pre-validation checkpoint saved (checkpoint_latest.pt)")
            
            # Validate
            val_metrics = self.validate(epoch)
            
            elapsed = time.time() - start_time
            
            # Log epoch summary
            val_fcs_str = f"{val_metrics['val_fcs']:.4f}" if "val_fcs" in val_metrics else "N/A"
            bleu_4_str = f"{val_metrics['bleu_4']:.4f}" if "bleu_4" in val_metrics else "N/A"
            rouge_l_str = f"{val_metrics['rouge_l']:.4f}" if "rouge_l" in val_metrics else "N/A"
            
            logger.info(
                f"Epoch {epoch+1}/{max_epochs} done in {elapsed:.1f}s | "
                f"Train Loss: {train_metrics['loss_total']:.4f} [Gen: {train_metrics['loss_generation']:.4f}, Consist: {train_metrics['loss_consistency']:.4f}] | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | Val FCS: {val_fcs_str} | "
                f"BLEU-4: {bleu_4_str} | ROUGE-L: {rouge_l_str}"
            )
            
            # Checkpoint saving decisions
            is_best = False
            if "val_fcs" in val_metrics:
                current_fcs = val_metrics["val_fcs"]
                current_bleu = val_metrics.get("bleu_4", 0.0)
                
                is_best_fcs = current_fcs > self.best_fcs
                if is_best_fcs:
                    self.best_fcs = current_fcs
                    
                is_best_bleu = current_bleu > self.best_bleu
                if is_best_bleu:
                    self.best_bleu = current_bleu
                    
                # We prioritize FCS (factual consistency) as our main checkpoint metric
                is_best = is_best_fcs
            
            # Save checkpoint state dicts
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": self.config,
            }
            
            save_checkpoint(
                state=checkpoint_state,
                checkpoint_dir=self.config.training.checkpoint_dir,
                filename=f"checkpoint_epoch_{epoch+1}.pt",
                is_best=is_best
            )
            
            # Early stopping check (only when heavy metrics are calculated)
            if "val_fcs" in val_metrics:
                early_stopping_metric = self.config.training.get("early_stopping_metric", "fcs")
                if early_stopping_metric == "fcs":
                    if is_best_fcs:
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                else:
                    if is_best_bleu:
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    logger.warning(
                        f"Early stopping triggered at epoch {epoch+1} due to no "
                        f"improvement in {early_stopping_metric} for {self.patience_counter} epochs."
                    )
                    break
