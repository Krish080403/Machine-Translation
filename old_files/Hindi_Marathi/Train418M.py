import os

try:
    train_command = 'CUDA_VISIBLE_DEVICES="0" python3.10 /home/krish/content/train.py /home/krish/content/Hindi_Marathi/wmt22_spm/wmt22_bin \
	   --arch transformer_wmt_en_de_big \
        --task translation_multi_simple_epoch \
        --finetune-from-model /home/krish/content/418M_last_checkpoint.pt \
        --save-dir /home/krish/content/Hindi_Marathi/checkpoint418M \
        --langs \'mr,en\' \
        --lang-pairs \'mr-en\' \
        --max-tokens 3600 \
        --encoder-normalize-before --decoder-normalize-before \
        --sampling-method temperature --sampling-temperature 1.5 \
        --encoder-langtok src --decoder-langtok \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
        --optimizer adam --adam-eps 1e-06 --adam-betas \'(0.9, 0.98)\' \
        --lr-scheduler inverse_sqrt --lr 3e-05 \
        --warmup-updates 2500 --max-update 40000 \
        --dropout 0.3 --attention-dropout 0.1 \
        --weight-decay 0.0 \
        --update-freq 2 --save-interval 5 \
        --save-interval-updates 5000 --keep-interval-updates 3 \
        --no-epoch-checkpoints \
        --seed 222 \
        --log-format simple \
        --log-interval 2 \
        --encoder-layers 12 --decoder-layers 12 \
        --encoder-layerdrop 0.05 --decoder-layerdrop 0.05 \
        --share-decoder-input-output-embed \
        --share-all-embeddings \
        --ddp-backend no_c10d'

    # Execute command
    os.system(train_command)

except Exception as e:
    print(f"Error: {e}")

