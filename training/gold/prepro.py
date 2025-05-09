import pandas as pd
import os

# Load the Excel files
train_df = pd.read_excel('train_gold.xlsx')
test_df = pd.read_excel('test.xlsx')
dev_df = pd.read_excel('validation.xlsx')

# Ensure directories exist
os.makedirs("wmt22", exist_ok=True)
os.makedirs("wmt22_spm", exist_ok=True)
os.makedirs("wmt22_spm/wmt22_bin", exist_ok=True)


# 🔹 Save merged translations
train_df['MUNDARI'].to_csv('wmt22/marathi_train.txt', index=False, header=False)
dev_df['MUNDARI'].to_csv('wmt22/marathi_dev.txt', index=False, header=False)
test_df['MUNDARI'].to_csv('wmt22/marathi_test.txt', index=False, header=False)

# 🔹 Save Hindi sentences
train_df['HINDI'].to_csv('wmt22/hindi_train.txt', index=False, header=False)
dev_df['HINDI'].to_csv('wmt22/hindi_dev.txt', index=False, header=False)
test_df['HINDI'].to_csv('wmt22/hindi_test.txt', index=False, header=False)

try:
    # SentencePiece encoding for Marathi
    marathi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=wmt22/marathi_train.txt --outputs=wmt22_spm/train.spm.mr'
    
    # SentencePiece encoding for Hindi
    hindi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=wmt22/hindi_train.txt --outputs=wmt22_spm/train.spm.hi'

    # Validation encoding
    dev_marathi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=wmt22/marathi_dev.txt --outputs=wmt22_spm/dev.spm.mr'
    dev_hindi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=wmt22/hindi_dev.txt --outputs=wmt22_spm/dev.spm.hi'

    # Test encoding
    test_marathi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=wmt22/marathi_test.txt --outputs=wmt22_spm/test.spm.mr'
    test_hindi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=wmt22/hindi_test.txt --outputs=wmt22_spm/test.spm.hi'

    # Execute commands
    os.system(marathi_command)
    os.system(hindi_command)
    os.system(dev_marathi_command)
    os.system(dev_hindi_command)
    os.system(test_marathi_command)
    os.system(test_hindi_command)

except Exception as e:
    print(f"Error: {e}")

try:
    # 🟢 Preprocessing for Fairseq
    preprocess_command = 'python3.10 /home/krish/content/preprocess.py \
        --source-lang hi \
        --target-lang mr \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir "wmt22_spm/wmt22_bin" \
        --srcdict "/home/krish/content/model_dict.128k.txt" \
        --tgtdict "/home/krish/content/model_dict.128k.txt" \
        --trainpref wmt22_spm/train.spm \
        --validpref wmt22_spm/dev.spm \
        --testpref wmt22_spm/test.spm'

    os.system(preprocess_command)

except Exception as e:
    print(f"Error: {e}")
