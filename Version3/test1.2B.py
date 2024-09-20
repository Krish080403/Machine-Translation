import os

try:
    # Command to generate translations
    generate_command = 'python3.10 /home/krish/content/Version3/modified_generate.py /home/krish/content/Version3/wmt22_spm/wmt22_bin --scoring chrf\
        --max-tokens 2000 \
        --batch-size 1 \
        --path /home/krish/content/Version3/checkpoint1.2B/checkpoint_best.pt \
        --fixed-dictionary /home/krish/content/Version3/wmt22_spm/model_dict.128k.txt \
        -s hi -t mr \
        --remove-bpe sentencepiece \
        --beam 5 \
        --task translation_multi_simple_epoch \
        --lang-pairs hi-mr \
        --decoder-langtok \
        --encoder-langtok src \
        --gen-subset test \
        --dataset-impl mmap \
        --distributed-world-size 1 \
        --distributed-no-spawn'

    # Execute command
    os.system(generate_command)

except Exception as e:
    print(f"Error: {e}")
