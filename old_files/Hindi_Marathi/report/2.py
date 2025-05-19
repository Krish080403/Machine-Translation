import os

try:
    test_marathi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=/home/krish/content/Hindi_Marathi/report/wmt22/marathi_test.txt --outputs=/home/krish/content/Hindi_Marathi/report/wmt22_spm/test.spm.mr'
    test_hindi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=/home/krish/content/Hindi_Marathi/report/wmt22/hindi_test.txt --outputs=/home/krish/content/Hindi_Marathi/report/wmt22_spm/test.spm.hi'
    os.system(test_marathi_command)
    os.system(test_hindi_command)

except Exception as e:
    print(f"Error: {e}")
