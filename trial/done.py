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
