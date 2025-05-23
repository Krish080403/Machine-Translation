import os

try:
    log_file='/home/krish/content/Hindi_Marathi/training_log.txt'
    train_command = (
        'CUDA_VISIBLE_DEVICES="1" fairseq-train /home/krish/content/Hindi_Nepali/wmt22_spm/wmt22_bin '
        '--finetune-from-model /home/krish/content/1.2B_last_checkpoint.pt '
        '--save-dir /home/krish/content/Hindi_Nepali/checkpoint1.2B '
        '--task translation_multi_simple_epoch '
        '--encoder-normalize-before '
        '--langs "hi,ne" '
        '--lang-pairs "hi-ne" '
        '--max-tokens 3600 '
        '--decoder-normalize-before '
        '--sampling-method temperature '
        '--sampling-temperature 1.5 '
        '--encoder-langtok src '
        '--decoder-langtok '
        '--criterion label_smoothed_cross_entropy '
        '--label-smoothing 0.2 '
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
        '--ddp-backend no_c10d'
	'| tee -a {log_file}'
    )
    # Execute command
    os.system(train_command)

except Exception as e:
    print(f"Error: {e}")
