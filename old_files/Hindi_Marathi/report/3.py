import os

try:
    preprocess_command = 'python3.10 /home/krish/content/preprocess.py \
        --source-lang mr \
        --target-lang hi \
        --testpref spm.$src.$tgt \
        --thresholdsrc 0 \
        --thresholdtgt 0 \
        --destdir "/home/krish/content/Hindi_Marathi/report/wmt22_spm/wmt22_bin" \
        --srcdict "/home/krish/content/Hindi_Marathi/wmt22_spm/model_dict.128k.txt" \
        --tgtdict "/home/krish/content/Hindi_Marathi/wmt22_spm/model_dict.128k.txt" \
        --testpref /home/krish/content/Hindi_Marathi/report/wmt22_spm/test.spm'

    # Execute command
    os.system(preprocess_command)

except Exception as e:
    print(f"Error: {e}")
