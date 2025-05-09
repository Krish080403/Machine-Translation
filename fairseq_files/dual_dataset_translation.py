from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq.data import LanguagePairDataset, data_utils, FairseqDataset
import os

class DualLanguagePairDataset(FairseqDataset):
    """
    A dataset wrapper that returns a pair of samples:
      - 'main': sample from the main LanguagePairDataset
      - 'second': sample from the second LanguagePairDataset
      
    It enforces that both datasets produce identical source inputs.
    """
    def __init__(self, main_dataset, second_dataset, third_dataset):
        self.main_dataset = main_dataset
        self.second_dataset = second_dataset
        self.third_dataset= third_dataset
        # Assert that the number of samples is the same
        assert len(main_dataset) == len(second_dataset), "Datasets must be the same length"
        assert len(main_dataset) == len(third_dataset)
        self._ordered_indices = None

    def __getitem__(self, index):
        return {
            "main": self.main_dataset[index],
            "second": self.second_dataset[index],
            "third" : self.third_dataset[index],
        }

    def __len__(self):
        return len(self.main_dataset)

    def collater(self, samples):
        main_samples = [s["main"] for s in samples if s is not None]
        second_samples = [s["second"] for s in samples if s is not None]
        third_samples = [s["third"] for s in samples if s is not None]
        return {
            "main": self.main_dataset.collater(main_samples),
            "second": self.second_dataset.collater(second_samples),
            "third": self.third_dataset.collater(third_samples),
        }

    def num_tokens(self, index):
        return self.main_dataset.num_tokens(index)

    def size(self, index):
        return self.main_dataset.size(index)

    def ordered_indices(self):
        if self._ordered_indices is None:
            self._ordered_indices = self.main_dataset.ordered_indices()
        return self._ordered_indices

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if hasattr(self.main_dataset, 'set_epoch'):
            self.main_dataset.set_epoch(epoch)
        if hasattr(self.second_dataset, 'set_epoch'):
            self.second_dataset.set_epoch(epoch)
        if hasattr(self.third_dataset, 'set_epoch'):
            self.third_dataset.set_epoch(epoch)
        self._ordered_indices = self.main_dataset.ordered_indices()
        self.second_dataset._ordered_indices = self._ordered_indices
        self.third_dataset._ordered_indices=self._ordered_indices
    def collater(self, samples):
        main_samples   = [s["main"]   for s in samples if s is not None]
        second_samples = [s["second"] for s in samples if s is not None]
        third_samples  = [s["third"]  for s in samples if s is not None]
        batch = {
            "main":   self.main_dataset.collater(main_samples),
            "second": self.second_dataset.collater(second_samples),
            "third":  self.third_dataset.collater(third_samples),
        }
        # inject epoch into the batch
        batch["epoch"] = getattr(self, "current_epoch", None)
        return batch

    def prefetch(self, indices):
        self.main_dataset.prefetch(indices)
        self.second_dataset.prefetch(indices)
        self.third_dataset.prefetch(indices)

    def batch_by_size(self, *args, **kwargs):
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
            help='Path to second dataset directory'
        )
        parser.add_argument(
            '--third-data-dir',
            type=str,
            required=True,
            help='Path to third dataset directory'
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        # Load the source and target dictionaries from the primary data directory.
        src_dict = cls.load_dictionary(os.path.join(args.data, f'dict.{args.source_lang}.txt'))
        tgt_dict = cls.load_dictionary(os.path.join(args.data, f'dict.{args.target_lang}.txt'))
        task = super().setup_task(args, **kwargs)
        task.src_dict = src_dict
        task.tgt_dict = tgt_dict
        return task

    def _load_language_pair_dataset(self, split, data_path):
        src, tgt = self.args.source_lang, self.args.target_lang
        prefix = os.path.join(data_path, f"{split}.{src}-{tgt}.")
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
        # Load the main dataset using the parent's loader from self.args.data.
        main_dataset = self._load_language_pair_dataset(split, self.args.data)

        # Load the second dataset from self.args.second_data_dir.
        second_path = os.path.join(self.args.second_data_dir, split)
        if os.path.isdir(second_path):
            second_dataset = self._load_language_pair_dataset(split, second_path)
        else:
            second_dataset = self._load_language_pair_dataset(split, self.args.second_data_dir)

        # Assert they have the same number of samples.
        assert len(main_dataset) == len(second_dataset), f"Mismatch between datasets in split '{split}'"
        second_dataset_aligned = LanguagePairDataset(
            src=main_dataset.src,              # Use the same source dataset object
            src_sizes=main_dataset.src_sizes,    # Use the same source sizes
            src_dict=self.src_dict,
            tgt=second_dataset.tgt,              # Keep the target from the second dataset
            tgt_sizes=second_dataset.tgt_sizes,
            tgt_dict=self.tgt_dict,
            left_pad_source=second_dataset.left_pad_source,
            left_pad_target=second_dataset.left_pad_target,
            input_feeding=second_dataset.input_feeding,
        )

        third_path = os.path.join(self.args.third_data_dir, split)
        if os.path.isdir(third_path):
            third_dataset = self._load_language_pair_dataset(split, third_path)
        else:
            third_dataset = self._load_language_pair_dataset(split, self.args.third_data_dir)

        # Assert they have the same number of samples.
        assert len(main_dataset) == len(third_dataset), f"Mismatch between datasets in split '{split}'"

        # Force the source in the second dataset to be identical to the main dataset.
        # Here, we create a new LanguagePairDataset for the second branch that reuses the source from the main dataset.
        third_dataset_aligned = LanguagePairDataset(
            src=main_dataset.src,              # Use the same source dataset object
            src_sizes=main_dataset.src_sizes,    # Use the same source sizes
            src_dict=self.src_dict,
            tgt=third_dataset.tgt,              # Keep the target from the second dataset
            tgt_sizes=third_dataset.tgt_sizes,
            tgt_dict=self.tgt_dict,
            left_pad_source=third_dataset.left_pad_source,
            left_pad_target=third_dataset.left_pad_target,
            input_feeding=third_dataset.input_feeding,
        )

        # Wrap both datasets into our DualLanguagePairDataset.
        self.datasets[split] = DualLanguagePairDataset(main_dataset, second_dataset_aligned,third_dataset_aligned)

    def dataset(self, split):
        return self.datasets[split]

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src=src_tokens,
            src_sizes=src_lengths,
            src_dict=self.src_dict,
            tgt=None,
            tgt_sizes=None,
            tgt_dict=self.tgt_dict,
        )
