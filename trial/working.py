from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq.data import LanguagePairDataset, data_utils
import os

@register_task("dual_dataset_translation")
class DualDatasetTranslationTask(TranslationMultiSimpleEpochTask):

    @classmethod
    def setup_task(cls, args, **kwargs):
        # Load dictionaries from the primary data directory
        src_dict = cls.load_dictionary(os.path.join(args.data, f"dict.{args.source_lang}.txt"))
        tgt_dict = cls.load_dictionary(os.path.join(args.data, f"dict.{args.target_lang}.txt"))
        task = super().setup_task(args, **kwargs)
        task.src_dict = src_dict
        task.tgt_dict = tgt_dict
        return task

    @classmethod
    def add_args(cls, parser):
        # Add standard arguments from the parent task
        super().add_args(parser)
        # Add argument for the second dataset directory
        parser.add_argument(
            '--second-data-dir',
            type=str,
            required=True,
            help='Path to the second dataset directory'
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        # Load the primary dataset using the parent method
        super().load_dataset(split, epoch=epoch, combine=combine, **kwargs)

        # Attempt to use a subdirectory named for the split in the second data directory
        data_path2 = os.path.join(self.args.second_data_dir, split)
        if not os.path.isdir(data_path2):
            # If such subdirectory does not exist, use the second-data-dir directly
            data_path2 = self.args.second_data_dir

        src, tgt = self.args.source_lang, self.args.target_lang
        # Construct prefix so that for split "valid" it becomes:
        #   "{data_path2}/valid.hi-mr." 
        # which expects files like "valid.hi-mr.hi.bin" and "valid.hi-mr.mr.bin"
        prefix2 = f"{data_path2}/{split}.{src}-{tgt}."

        # Load the indexed dataset for the source language from the second data directory
        src_dataset = data_utils.load_indexed_dataset(prefix2 + src, self.src_dict, self.args.dataset_impl)
        if src_dataset is None:
            raise FileNotFoundError(f"Indexed dataset file not found: {prefix2 + src}(.bin/.idx)")
        
        # Load the indexed dataset for the target language from the second data directory
        tgt_dataset = data_utils.load_indexed_dataset(prefix2 + tgt, self.tgt_dict, self.args.dataset_impl)
        if tgt_dataset is None:
            raise FileNotFoundError(f"Indexed dataset file not found: {prefix2 + tgt}(.bin/.idx)")

        # Create a LanguagePairDataset for the second dataset
        second_dataset = LanguagePairDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict
        )

        # Store both datasets (primary and second) for later use (e.g., by a custom criterion)
        self.datasets[split] = {
            "main": self.datasets[split],
            "second": second_dataset
        }

    def dataset(self, split):
        return self.datasets[split]

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        # For inference, we only use the primary dataset's source tokens
        return LanguagePairDataset(
            src_tokens, src_lengths, self.src_dict,
            tgt=None, tgt_sizes=None, tgt_dict=self.tgt_dict
        )
