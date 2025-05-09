from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.criterions import register_criterion
import torch
from fairseq.logging import metrics
import torch.nn.functional as F

@register_criterion("dual_label_smoothed_cross_entropy")
class DualLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    """
    This criterion computes three independent losses:
    1. Cross-entropy loss on the main branch
    2. Cosine similarity loss on the second branch
    3. Cosine similarity loss on the third branch
    
    It then combines them as:
         total_loss = alpha * loss1 + beta * loss2 + gamma * loss3
         
    For deterministic behavior (when the inputs are supposed to be identical),
    we fix the random seed for all forward passes.
    """
    def __init__(self, task, sentence_avg, label_smoothing, alpha, beta, gamma):
        super().__init__(task, sentence_avg, label_smoothing)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Get padding index from the task's dictionary
        self.padding_idx = task.target_dictionary.pad()

    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--alpha', type=float, default=1.0,
                            help='Weight for loss computed on the main dataset')
        parser.add_argument('--beta', type=float, default=1.0,
                            help='Weight for loss computed on the second dataset')
        parser.add_argument('--gamma', type=float, default=1.0,
                            help='Weight for loss computed on the third dataset')

    def forward(self, model, sample, reduce=True):
        """
        Args:
            model: The model to evaluate.
            sample: A dictionary with keys "main", "second" and "third" each a sample dict.
            reduce: Whether to reduce the loss (as in standard Fairseq criteria).
        Returns:
            total_loss, sample_size, logging_output
            
        """
        epoch = sample.get("epoch", None)
        if epoch is not None:
            # Log the current epoch
            metrics.log_scalar("epoch", epoch, sample["main"]["ntokens"])
            
        # Save the current RNG state to restore later
        state = torch.get_rng_state()
        # Set a fixed seed for determinism between forward passes
        fixed_seed = 12345 + epoch if epoch is not None else 12345

        # Run forward pass for the main branch with fixed seed
        torch.manual_seed(fixed_seed)
        net_input_main = sample["main"]["net_input"]
        net_output_main = model(**net_input_main)
        # Standard cross-entropy loss
        loss1, nll_loss1 = self.compute_loss(model, net_output_main, sample["main"], reduce=reduce)

        # Get the embedding matrix from decoder
        embed_mat = model.decoder.embed_tokens.weight  # [V, H]
        self.padding_idx = model.decoder.padding_idx

        # Reset the RNG state and set the fixed seed again before the second forward pass
        torch.set_rng_state(state)
        torch.manual_seed(fixed_seed)

        #### SECOND branch: cosine similarity loss with incorrect translations ####
        net_input_second = sample["second"]["net_input"]
        net_output_second = model(**net_input_second)
        logits2 = net_output_second[0]                    # [B, T, V]
        probs2 = F.softmax(logits2, dim=-1)               # [B, T, V]
        pred_emb2 = probs2.matmul(embed_mat)              # [B, T, H]
        tgt_tokens2 = sample["second"]["target"]          # [B, T]
        gold_emb2 = model.decoder.embed_tokens(tgt_tokens2)  # [B, T, H]
        cos_sim2 = F.cosine_similarity(pred_emb2, gold_emb2, dim=-1)  # [B, T]
        
        # Create padding mask
        padding_mask2 = tgt_tokens2.eq(self.padding_idx)
        
        # Check for sentences that are all padding (blank targets)
        # Sum over time dimension (T) to get count of padding tokens per sentence
        padding_count = padding_mask2.sum(dim=1)  # [B]
        # Get sequence length for each sentence in the batch
        seq_lengths = tgt_tokens2.size(1) * torch.ones_like(padding_count)  # [B]
        # Identify sentences that are all padding (blank targets)
        blank_sentence_mask = (padding_count == seq_lengths)  # [B]
        
        # Extend blank_sentence_mask to match padding_mask2 dimensions
        expanded_blank_mask = blank_sentence_mask.unsqueeze(1).expand_as(padding_mask2)  # [B, T]
        
        # Create a combined mask that includes both padding tokens and blank sentences
        combined_mask2 = padding_mask2 | expanded_blank_mask  # [B, T]
        
        # Count valid (non-padding and non-blank) elements
        valid_elements2 = (~combined_mask2).float().sum()
        
        if valid_elements2 > 0:
            # Mask out padding tokens and blank sentences in cosine similarity calculation
            masked_sim2 = cos_sim2.masked_fill(combined_mask2, 0.0)
            # Compute average cosine similarity
            sim_score2 = masked_sim2.sum() / valid_elements2
            # Compute loss2 = β * (average_cosine_similarity)
            # We want to minimize similarity with incorrect translations
            loss2 = self.beta * (sim_score2)
        else:
            # If all sentences are blank, set loss2 to 0
            loss2 = torch.tensor(0.0, device=logits2.device)

        # Reset the RNG state and set the fixed seed again before the third forward pass
        torch.set_rng_state(state)
        torch.manual_seed(fixed_seed)
        
        #### THIRD branch: cosine similarity loss with correct translations ####
        net_input_third = sample["third"]["net_input"]
        net_output_third = model(**net_input_third)
        logits3 = net_output_third[0]                     # [B, T, V]
        probs3 = F.softmax(logits3, dim=-1)               # [B, T, V]
        pred_emb3 = probs3.matmul(embed_mat)              # [B, T, H]
        tgt_tokens3 = sample["third"]["target"]           # [B, T]
        gold_emb3 = model.decoder.embed_tokens(tgt_tokens3)  # [B, T, H]
        cos_sim3 = F.cosine_similarity(pred_emb3, gold_emb3, dim=-1)  # [B, T]
        
        # Create padding mask
        padding_mask3 = tgt_tokens3.eq(self.padding_idx)
        
        # Check for sentences that are all padding (blank targets)
        # Sum over time dimension (T) to get count of padding tokens per sentence
        padding_count = padding_mask3.sum(dim=1)  # [B]
        # Get sequence length for each sentence in the batch
        seq_lengths = tgt_tokens3.size(1) * torch.ones_like(padding_count)  # [B]
        # Identify sentences that are all padding (blank targets)
        blank_sentence_mask = (padding_count == seq_lengths)  # [B]
        
        # Extend blank_sentence_mask to match padding_mask3 dimensions
        expanded_blank_mask = blank_sentence_mask.unsqueeze(1).expand_as(padding_mask3)  # [B, T]
        
        # Create a combined mask that includes both padding tokens and blank sentences
        combined_mask3 = padding_mask3 | expanded_blank_mask  # [B, T]
        
        # Count valid (non-padding and non-blank) elements
        valid_elements3 = (~combined_mask3).float().sum()
        
        if valid_elements3 > 0:
            # Mask out padding tokens and blank sentences in cosine similarity calculation
            masked_sim3 = cos_sim3.masked_fill(combined_mask3, 0.0)
            # Compute average cosine similarity
            sim_score3 = masked_sim3.sum() / valid_elements3
            # Compute loss3 = γ * (1 / average_cosine_similarity)
            # We want to maximize similarity with correct translations
            eps = 1e-6  # Small epsilon to avoid division by zero
            loss3 = self.gamma * (1.0 / (sim_score3 + eps))
        else:
            # If all sentences are blank, set loss3 to 0
            loss3 = torch.tensor(0.0, device=logits3.device)

        # Restore original RNG state so that subsequent operations are unaffected
        torch.set_rng_state(state)

        # Combine all three losses with their respective weights
        total_loss = self.alpha * loss1 + loss2 + loss3

        if self.sentence_avg:
            sample_size = sample["main"]["target"].size(0)
        else:
            sample_size = sample["main"]["ntokens"]

        logging_output = {
            "loss": total_loss.data,
            "loss1": loss1.data,
            "loss2": loss2.data,
            "loss3": loss3.data,
            "nll_loss": nll_loss1.data,
            "ntokens": sample["main"]["ntokens"],
            "nsentences": sample["main"]["target"].size(0),
            "sample_size": sample_size,
        }
        return total_loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss1_sum = sum(log.get("loss1", 0) for log in logging_outputs)
        loss2_sum = sum(log.get("loss2", 0) for log in logging_outputs)
        loss3_sum = sum(log.get("loss3", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("loss1", loss1_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("loss2", loss2_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("loss3", loss3_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("nll_loss", nll_loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("ppl", 2 ** (nll_loss_sum / ntokens), ntokens, round=2)