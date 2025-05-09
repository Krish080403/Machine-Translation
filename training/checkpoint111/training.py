import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    fairseq_path = '/home/krish/content/fairseq'
    log_file = 'log.txt'
    train_command = (
        f'CUDA_VISIBLE_DEVICES="2" PYTHONPATH={fairseq_path} '  
        f'python3.10 {fairseq_path}/fairseq_cli/train.py /home/krish/content/training/gold/wmt22_spm/wmt22_bin '
        '--second-data-dir /home/krish/content/training/incorrect/wmt22_spm/wmt22_bin '
        '--third-data-dir /home/krish/content/training/correct/wmt22_spm/wmt22_bin '
        '--finetune-from-model /home/krish/content/1.2B_last_checkpoint.pt '
        '--save-dir checkpoint1.2B '
        '--task dual_dataset_translation '
        '--encoder-normalize-before '
        '--source-lang "hi" '
        '--target-lang "mr" '
        '--langs "hi","mr" '
        '--lang-pairs "hi-mr" '
        '--max-tokens 1000 '
        '--decoder-normalize-before '
        '--sampling-method temperature '
        '--sampling-temperature 1.5 '
        '--encoder-langtok src '
        '--decoder-langtok '
        '--criterion dual_label_smoothed_cross_entropy '
        '--alpha 1 '
        '--beta 1 '
        '--gamma 1 '
        '--optimizer adam '
        '--adam-eps 1e-06 '
        '--adam-betas "(0.9, 0.98)" '
        '--lr-scheduler inverse_sqrt '
        '--lr 3e-05 '
        '--warmup-updates 2500 '
        '--max-update 40000 '
        '--dropout 0.3 '
        '--attention-dropout 0.1 '
        '--weight-decay 0.0 '
        '--update-freq 2 '
        '--save-interval 1 '
        '--save-interval-updates 5000 '
        '--keep-interval-updates 10 '
        '--no-epoch-checkpoints '
        '--seed 222 '
        '--log-format simple '
        '--log-interval 2 '
        '--patience 10 '
        '--arch transformer_wmt_en_de_big '
        '--encoder-layers 24 '
        '--decoder-layers 24 '
        '--encoder-ffn-embed-dim 8192 '
        '--decoder-ffn-embed-dim 8192 '
        '--encoder-layerdrop 0.05 '
        '--decoder-layerdrop 0.05 '
        '--share-decoder-input-output-embed '
        '--share-all-embeddings '
        '--ddp-backend no_c10d '
        f'| tee -a {log_file}'
    )

    logger.info("Starting training with extra loss...")
    logger.info(f"Training command: {train_command}")

    # Execute command
    os.system(train_command)
    logger.info("Training completed.")

except Exception as e:
    logger.error(f"Error during training: {e}")