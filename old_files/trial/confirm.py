from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq.data import LanguagePairDataset, data_utils, FairseqDataset
import os

class DualLanguagePairDataset(FairseqDataset):
    """
    A dataset that returns paired samples: one from the 'main' dataset and one from the 'second',
    ensuring both use the same ordered indices and thus produce identical batches if data is the same.
    """
    def __init__(self, main_dataset, second_dataset):
        self.main_dataset = main_dataset
        self.second_dataset = second_dataset
        assert len(main_dataset) == len(second_dataset), "Datasets must be the same length"
        self._ordered_indices = None

    def __getitem__(self, index):
        return {
            "main": self.main_dataset[index],
            "second": self.second_dataset[index],
        }

    def __len__(self):
        return len(self.main_dataset)

    def collater(self, samples):
        # Collate main and second samples separately
        main_samples = [s["main"] for s in samples if s is not None]
        second_samples = [s["second"] for s in samples if s is not None]
        return {
            "main": self.main_dataset.collater(main_samples),
            "second": self.second_dataset.collater(second_samples),
        }

    def num_tokens(self, index):
        # same source sizes => use the main dataset's token count
        return self.main_dataset.num_tokens(index)

    def size(self, index):
        return self.main_dataset.size(index)

    def ordered_indices(self):
        # If not set, get from main_dataset
        if self._ordered_indices is None:
            self._ordered_indices = self.main_dataset.ordered_indices()
        return self._ordered_indices

    def set_epoch(self, epoch):
        # Let each dataset set their epoch
        if hasattr(self.main_dataset, 'set_epoch'):
            self.main_dataset.set_epoch(epoch)
        if hasattr(self.second_dataset, 'set_epoch'):
            self.second_dataset.set_epoch(epoch)
        # Force shared ordering
        self._ordered_indices = self.main_dataset.ordered_indices()
        # If the second dataset is a LanguagePairDataset (or a subclass), we can unify ordering:
        if hasattr(self.second_dataset, '_ordered_indices'):
            self.second_dataset._ordered_indices = self._ordered_indices

    def prefetch(self, indices):
        self.main_dataset.prefetch(indices)
        self.second_dataset.prefetch(indices)

    def batch_by_size(self, *args, **kwargs):
        # Return the main dataset's batch sampler so that both are identical
        return self.main_dataset.batch_by_size(*args, **kwargs)


@register_task("dual_dataset_translation")
class DualDatasetTranslationTask(TranslationMultiSimpleEpochTask):

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument(
            '--second-data-dir',
            type=str,
            required=True,
            help='Path to the second dataset directory'
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        # Load dictionaries from the primary data directory
        src_dict = cls.load_dictionary(os.path.join(args.data, f'dict.{args.source_lang}.txt'))
        tgt_dict = cls.load_dictionary(os.path.join(args.data, f'dict.{args.target_lang}.txt'))

        task = super().setup_task(args, **kwargs)
        task.src_dict = src_dict
        task.tgt_dict = tgt_dict
        return task

    def _load_language_pair_dataset(self, split, data_path):
        src, tgt = self.args.source_lang, self.args.target_lang
        prefix = os.path.join(data_path, f"{split}.{src}-{tgt}.")
        # Load source & target
        src_dataset = data_utils.load_indexed_dataset(prefix + src, self.src_dict, self.args.dataset_impl)
        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, self.tgt_dict, self.args.dataset_impl)

        if src_dataset is None or tgt_dataset is None:
            raise FileNotFoundError(f"Could not find dataset: {prefix + src} or {prefix + tgt}")

        return LanguagePairDataset(
            src=src_dataset,
            src_sizes=src_dataset.sizes,
            src_dict=self.src_dict,
            tgt=tgt_dataset,
            tgt_sizes=tgt_dataset.sizes,
            tgt_dict=self.tgt_dict,
            left_pad_source=False,
            left_pad_target=False,
            input_feeding=True,
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        # Load main dataset from self.args.data
        main_dataset = self._load_language_pair_dataset(split, self.args.data)

        # Load second dataset from self.args.second_data_dir
        second_path = os.path.join(self.args.second_data_dir, split)
        if os.path.isdir(second_path):
            second_dataset = self._load_language_pair_dataset(split, second_path)
        else:
            second_dataset = self._load_language_pair_dataset(split, self.args.second_data_dir)

        # Ensure same length
        assert len(main_dataset) == len(second_dataset), f"Mismatch between datasets in split '{split}'"

        # Wrap both in the DualLanguagePairDataset
        self.datasets[split] = DualLanguagePairDataset(main_dataset, second_dataset)

    def dataset(self, split):
        return self.datasets[split]

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        # Standard inference dataset
        return LanguagePairDataset(
            src=src_tokens,
            src_sizes=src_lengths,
            src_dict=self.src_dict,
            tgt=None,
            tgt_sizes=None,
            tgt_dict=self.tgt_dict,
        )






##CRITERION
##CRITERION
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.criterions import register_criterion
import torch
from fairseq.logging import metrics

@register_criterion("dual_label_smoothed_cross_entropy")
class DualLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    """
    This criterion computes two independent cross-entropy losses on the 'main' and 'second' branches.
    It then combines them as:
         total_loss = alpha * loss1 + beta * loss2
    For deterministic behavior (when the inputs are supposed to be identical),
    we fix the random seed for both forward passes.
    """
    def __init__(self, task, sentence_avg, label_smoothing, alpha, beta,gamma):
        super().__init__(task, sentence_avg, label_smoothing)
        self.alpha = alpha
        self.beta = beta
        self.gamma=gamma

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
        # Save the current RNG state to restore later
        state = torch.get_rng_state()
        # Set a fixed seed for determinism between forward passes
        fixed_seed = 12345

        # Run forward pass for the main branch with fixed seed
        torch.manual_seed(fixed_seed)
        net_input_main = sample["main"]["net_input"]
        net_output_main = model(**net_input_main)
        loss1, nll_loss1 = self.compute_loss(model, net_output_main, sample["main"], reduce=reduce)

        # Reset the RNG state and set the fixed seed again before the second forward pass
        torch.set_rng_state(state)
        torch.manual_seed(fixed_seed)
        net_input_second = sample["second"]["net_input"]
        net_output_second = model(**net_input_second)
        loss2, nll_loss2 = self.compute_loss(model, net_output_second, sample["second"], reduce=reduce)

        #third data
        torch.set_rng_state(state)
        torch.manual_seed(fixed_seed)
        net_input_third = sample["third"]["net_input"]
        net_output_third = model(**net_input_third)
        loss3, nll_loss3 = self.compute_loss(model, net_output_third, sample["third"], reduce=reduce)
        # Restore original RNG state so that subsequent operations are unaffected
        torch.set_rng_state(state)

        total_loss = self.alpha * loss1 + self.beta * loss2 + self.gamma*loss3

        if self.sentence_avg:
            sample_size = sample["main"]["target"].size(0)
        else:
            sample_size = sample["main"]["ntokens"]

        logging_output = {
            "loss": total_loss.data,
            "loss1": loss1.data,
            "loss2": loss2.data,
            "loss3": loss3.data,
            "nll_loss": (nll_loss1 + nll_loss2+nll_loss3).data,
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
        # Save the current RNG state to restore later
        state = torch.get_rng_state()
        # Set a fixed seed for determinism between forward passes
        fixed_seed = 12345

        # Run forward pass for the main branch with fixed seed
        torch.manual_seed(fixed_seed)
        net_input_main = sample["main"]["net_input"]
        net_output_main = model(**net_input_main)
        loss1, nll_loss1 = self.compute_loss(model, net_output_main, sample["main"], reduce=reduce)

        # Get the embedding matrix from decoder
        embed_mat = model.decoder.embed_tokens.weight  # [V, H]
        self.padding_idx = model.decoder.padding_idx

        # Reset the RNG state and set the fixed seed again before the second forward pass
        torch.set_rng_state(state)
        torch.manual_seed(fixed_seed)

        #### SECOND branch: cosine‐sim loss ####
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
        # Count valid (non-padding) elements
        valid_elements2 = (~padding_mask2).float().sum()
        # Mask out padding tokens in cosine similarity calculation
        masked_sim2 = cos_sim2.masked_fill(padding_mask2, 0.0)
        # Compute loss only on non-padding tokens
        loss2 = (1.0 - masked_sim2).sum() / valid_elements2

        # Reset the RNG state and set the fixed seed again before the third forward pass
        torch.set_rng_state(state)
        torch.manual_seed(fixed_seed)
        
        #### THIRD branch: cosine‐sim loss ####
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
        # Count valid (non-padding) elements
        valid_elements3 = (~padding_mask3).float().sum()
        # Mask out padding tokens in cosine similarity calculation
        masked_sim3 = cos_sim3.masked_fill(padding_mask3, 0.0)
        # Compute loss only on non-padding tokens
        loss3 = (1.0 - masked_sim3).sum() / valid_elements3

        # Restore original RNG state so that subsequent operations are unaffected
        torch.set_rng_state(state)

        # Combine all three losses with their respective weights
        total_loss = self.alpha * loss1 + self.beta * loss2 + self.gamma * loss3

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