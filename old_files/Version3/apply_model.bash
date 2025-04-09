#Train hi
python3.10 /home/krish/content/fairseq/scripts/spm_encode.py \
    --model spm.128k.model \
    --output_format=piece \
    --inputs=wmt22/hindi_train.txt \
    --outputs=wmt22_spm/train.spm.hi
#Train mr
python3.10 /home/krish/content/fairseq/scripts/spm_encode.py \
    --model spm.128k.model \
    --output_format=piece \
    --inputs=wmt22/marathi_train.txt \
    --outputs=wmt22_spm/train.spm.mr
#Dev hi
python3.10 /home/krish/content/fairseq/scripts/spm_encode.py \
    --model spm.128k.model \
    --output_format=piece \
    --inputs=wmt22/hindi_dev.txt \
    --outputs=wmt22_spm/dev.spm.hi
#Dev mr
python3.10 /home/krish/content/fairseq/scripts/spm_encode.py \
    --model spm.128k.model \
    --output_format=piece \
    --inputs=wmt22/marathi_dev.txt \
    --outputs=wmt22_spm/dev.spm.mr
#Test hi
python3.10 /home/krish/content/fairseq/scripts/spm_encode.py \
    --model spm.128k.model \
    --output_format=piece \
    --inputs=wmt22/hindi_test.txt \
    --outputs=wmt22_spm/test.spm.hi
#Test mr
python3.10 /home/krish/content/fairseq/scripts/spm_encode.py \
    --model spm.128k.model \
    --output_format=piece \
    --inputs=wmt22/marathi_test.txt \
    --outputs=wmt22_spm/test.spm.mr