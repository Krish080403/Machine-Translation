import os

try:
    # Command to generate translations and redirect output to out.txt
    generate_command = (
        'python3.10 /home/krish/content/gradio/modified_generate.py /home/krish/content/gradio '
        '--max-tokens 2000 '
        '--batch-size 1 '
        '--path /home/krish/content/Hindi_Marathi/new_checkpoint1.2B_Mr-hi/checkpoint_best.pt '
        '--fixed-dictionary /home/krish/content/Hindi_Marathi/wmt22_spm/model_dict.128k.txt '
        '-s mr -t hi '
        '--remove-bpe sentencepiece '
        '--beam 5 '
        '--task translation_multi_simple_epoch '
        '--lang-pairs mr-hi '
        '--decoder-langtok '
        '--encoder-langtok src '
        '--gen-subset test '
        '--dataset-impl mmap '
        '--distributed-world-size 1 '
        '--distributed-no-spawn '
        '> output.txt'  # Redirect output to out.txt
    )

    # Execute the command
    os.system(generate_command)

except Exception as e:
    print(f"An error occurred: {str(e)}")
