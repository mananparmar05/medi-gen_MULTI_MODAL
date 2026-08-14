"""
Beam Search Decoding for Multimodal Medical Report Generation.

Replaces greedy decoding with a width-B beam search that:
    1. Maintains `num_beams` candidate sequences simultaneously.
    2. Applies length penalty to prefer complete sentences over short ones.
    3. Applies repetition penalty to suppress repeated tokens.
    4. Blocks repeated n-grams (no_repeat_ngram_size).
    5. Enforces minimum output length to prevent empty reports.

Usage (from report_generator.py generate()):
    from utils.beam_search import BeamSearchDecoder
    searcher = BeamSearchDecoder(num_beams=4, ...)
    generated_ids, hidden_states = searcher.search(decoder, visual_features, tokenizer, max_length)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BeamHypothesis:
    """A single beam hypothesis — a partially generated token sequence."""
    token_ids: List[int] = field(default_factory=list)
    score: float = 0.0          # cumulative log-probability (length-normalised at end)
    is_done: bool = False


class BeamSearchDecoder:
    """
    Beam search decoder for autoregressive medical report generation.

    Args:
        num_beams: Number of beams to maintain (default 4).
            Higher = better quality but slower. 4 is a good balance on CPU.
        length_penalty: Exponent for length normalisation.
            > 1.0 favours longer sequences; < 1.0 favours shorter.
            0.6 is standard for translation/summarisation tasks.
        repetition_penalty: Logit divisor for previously seen tokens (>1.0).
            1.3 is a reasonable default for radiology reports.
        no_repeat_ngram_size: Block repeated n-grams of this size (default 3).
        min_length: Suppress EOS for this many steps (default 10).
        max_length: Hard maximum output length.
    """

    def __init__(
        self,
        num_beams: int = 4,
        length_penalty: float = 0.6,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
        min_length: int = 10,
        max_length: int = 128,
    ):
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.min_length = min_length
        self.max_length = max_length

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        token_ids: List[int],
    ) -> torch.Tensor:
        """Divide logits of already-generated tokens by repetition_penalty."""
        if self.repetition_penalty == 1.0:
            return logits
        for token_id in set(token_ids):
            if logits[token_id] > 0:
                logits[token_id] /= self.repetition_penalty
            else:
                logits[token_id] *= self.repetition_penalty
        return logits

    def _apply_ngram_blocking(
        self,
        logits: torch.Tensor,
        token_ids: List[int],
    ) -> torch.Tensor:
        """Set logits of tokens that would complete a repeated n-gram to -inf."""
        n = self.no_repeat_ngram_size
        if n < 2 or len(token_ids) < n:
            return logits
        prefix = tuple(token_ids[-(n - 1):])
        banned = set()
        for i in range(len(token_ids) - n + 1):
            if tuple(token_ids[i:i + n - 1]) == prefix:
                banned.add(token_ids[i + n - 1])
        for token_id in banned:
            logits[token_id] = -float("inf")
        return logits

    @torch.no_grad()
    def search(
        self,
        decoder,
        visual_features: torch.Tensor,
        tokenizer,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run beam search over the decoder for a single sample (batch size = 1).

        Args:
            decoder: MedicalReportDecoder instance.
            visual_features: [1, C, H, W] visual context for one sample.
            tokenizer: ReportTokenizer instance.
            max_length: Override max generation length.

        Returns:
            (best_ids_tensor, dummy_hidden_states)
            best_ids_tensor: [1, gen_len] best beam token IDs.
            dummy_hidden_states: [1, gen_len, 768] hidden states of best beam
                                 (collected during decoding for cross-attn bridge).
        """
        if max_length is None:
            max_length = self.max_length

        device = visual_features.device
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

        # --- Initialise beams ---
        # Each beam: (token_ids list, cumulative log-prob, done flag, hidden_states list)
        beams: List[Tuple[List[int], float, bool, List[torch.Tensor]]] = [
            ([bos_id], 0.0, False, [])
        ]
        completed_beams: List[Tuple[List[int], float]] = []

        for step in range(max_length - 1):
            # Collect all next-step candidates across all active beams
            all_candidates: List[Tuple[List[int], float, bool, List[torch.Tensor]]] = []

            for token_ids, cum_log_prob, is_done, hidden_list in beams:
                if is_done:
                    all_candidates.append((token_ids, cum_log_prob, True, hidden_list))
                    continue

                # Forward pass through decoder for this beam
                input_ids = torch.tensor(
                    [token_ids], dtype=torch.long, device=device
                )
                outputs = decoder.forward(
                    visual_features=visual_features,
                    input_ids=input_ids,
                )

                # Logits for the last position: [vocab_size]
                next_logits = outputs["logits"][0, -1, :].clone()
                last_hidden = outputs["hidden_states"][0:1, -1:, :]  # [1, 1, 768]

                # Apply penalties
                next_logits = self._apply_repetition_penalty(next_logits, token_ids)
                next_logits = self._apply_ngram_blocking(next_logits, token_ids)

                # Suppress EOS for minimum length
                if step < self.min_length:
                    next_logits[eos_id] = -float("inf")

                # Convert to log-probabilities
                log_probs = F.log_softmax(next_logits, dim=-1)

                # Take top-k next tokens
                topk_log_probs, topk_ids = log_probs.topk(self.num_beams)

                for log_p, token_id in zip(
                    topk_log_probs.tolist(), topk_ids.tolist()
                ):
                    new_ids = token_ids + [token_id]
                    new_score = cum_log_prob + log_p
                    new_hidden = hidden_list + [last_hidden]
                    done = token_id == eos_id
                    all_candidates.append((new_ids, new_score, done, new_hidden))

            # --- Length-normalised beam selection ---
            def normalised_score(item):
                ids, score, done, _ = item
                length = max(len(ids), 1)
                return score / (length ** self.length_penalty)

            all_candidates.sort(key=normalised_score, reverse=True)

            # Separate completed vs. active beams
            beams = []
            for candidate in all_candidates:
                ids, score, done, hidden_list = candidate
                if done:
                    completed_beams.append((ids, normalised_score(candidate)))
                    if len(completed_beams) >= self.num_beams:
                        break
                elif len(beams) < self.num_beams:
                    beams.append(candidate)

            # Early exit if we have enough completed beams
            if len(completed_beams) >= self.num_beams:
                break

            # If all remaining beams are done, stop
            if not beams:
                break

        # --- Pick best completed beam, or fall back to best active beam ---
        if completed_beams:
            best_ids, _ = max(completed_beams, key=lambda x: x[1])
            # Recover hidden states from best active beam (closest match by prefix)
            best_hidden_list = beams[0][3] if beams else []
        else:
            # No beam completed with EOS — take the highest-scoring active beam
            best_beam = max(beams, key=lambda b: normalised_score(b))
            best_ids = best_beam[0]
            best_hidden_list = best_beam[3]

        # Stack hidden states; pad if necessary for cross-attention bridge
        if best_hidden_list:
            hidden_states_all = torch.cat(best_hidden_list, dim=1)  # [1, gen_len, 768]
        else:
            # Fallback: zero tensor
            hidden_states_all = torch.zeros(1, len(best_ids), decoder.hidden_dim, device=device)

        best_ids_tensor = torch.tensor([best_ids], dtype=torch.long, device=device)

        return best_ids_tensor, hidden_states_all
