import subprocess
import os
import time
import pandas as pd
from IPython.display import display

def translate_text(input_file, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

    try:
        spm_encode_command_mr = (
            f'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model '
            f'--output_format=piece --inputs={input_file} --outputs=/home/krish/content/backtranslation/backdata/test.spm.mr'
        )
        os.system(spm_encode_command_mr)

        spm_encode_command_hi = (
            f'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model '
            f'--output_format=piece --inputs={input_file} --outputs=/home/krish/content/backtranslation/backdata/test.spm.hi'
        )
        os.system(spm_encode_command_hi)

        preprocess_command = (
            f'python3.10 /home/krish/content/preprocess.py '
            f'--source-lang mr '
            f'--target-lang hi '
            f'--testpref /home/krish/content/backtranslation/backdata/test.spm '
            f'--thresholdsrc 0 '
            f'--thresholdtgt 0 '
            f'--destdir "/home/krish/content/backtranslation/backdata" '
            f'--srcdict "/home/krish/content/Hindi_Marathi/wmt22_spm/model_dict.128k.txt" '
            f'--tgtdict "/home/krish/content/Hindi_Marathi/wmt22_spm/model_dict.128k.txt"'
        )
        os.system(preprocess_command)

        generate_command = (
            'python3.10 /home/krish/content/backtranslation/modified_generate.py /home/krish/content/backtranslation/backdata '
            '--max-tokens 2000 '
            '--batch-size 1 '
            '--path /home/krish/content/Hindi_Marathi/new_checkpoint1.2B_Mr-Hi/checkpoint_best.pt '
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
            f'>out.txt '
        )
        os.system(generate_command)
        dehypo_sentences = []
        sample_id=[]
        source_sentences = []
        target_sentences = []
        hypo_sentences = []
        chrf_scores = []
        chrf_plus_scores=[]
        bleu_scores=[]
        bleu_details=[]
        Log_Probability=[]
        with open('out.txt', 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith("D-"):
                    dehypo_sentence = line[len("D- "):]
                    dehypo_sentences.append(dehypo_sentence)
                if line.startswith("Sample_Id-"):
                    sample = line[len("Sample_Id-"):]
                    sample_id.append(sample)
                elif line.startswith("Cfhrf score:"):
                    chrf_score = line[len("Cfhrf score: "):]
                    chrf_scores.append(chrf_score)
                elif line.startswith("Cfhrf++ score:"):
                    chrf_plus_score = line[len("Cfhrf++ score: "):]
                    chrf_plus_scores.append(chrf_plus_score)
                elif line.startswith("Bleu score: "):
                    bleu_detail = line[len("Bleu score: "):]
                    bleu_details.append(bleu_detail)
        # Create a DataFrame
        data = {
            'Sample_id': sample_id,
            'Generated': dehypo_sentences,
            'CHRF2': chrf_scores,
            'CHRF2++': chrf_plus_scores,
            'BLEU Details': bleu_details,

        }
        df = pd.DataFrame(data)
        df['Sample_id'] = pd.to_numeric(df['Sample_id'], errors='coerce')
        df= df.sort_values(by='Sample_id')
        df= df.reset_index(drop=True)
        df.sort_values(by='Sample_id',inplace=True)
        df=df.drop(columns='Sample_id')
        display(df)
        df.to_csv(output_file, sep='\t', index=False,header=True)

        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                translated_text = f.read()
            return translated_text
        else:
            return "Output file not found. Translation may have failed."

    except Exception as e:
        return f"An error occurred: {e}"
def main():
    input_file = "output.txt"
    output_file = "backtranslated.txt"
    result = translate_text(input_file, output_file)
    print(result)

if __name__ == "__main__":
    main()
