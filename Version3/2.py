import os

try:
    # Command to train marathi
    marathi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=/home/krish/content/Version3/wmt22/marathi_train.txt --outputs=/home/krish/content/Version3/wmt22_spm/train.spm.mr'
    
    # Command to train Hindi
    hindi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=/home/krish/content/Version3/wmt22/hindi_train.txt --outputs=/home/krish/content/Version3/wmt22_spm/train.spm.hi'

    # Command to validate marathi
    dev_marathi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=/home/krish/content/Version3/wmt22/marathi_dev.txt --outputs=/home/krish/content/Version3/wmt22_spm/dev.spm.mr'

    # Command to validate Hindi
    dev_hindi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=/home/krish/content/Version3/wmt22/hindi_dev.txt --outputs=/home/krish/content/Version3/wmt22_spm/dev.spm.hi'

    # Command to test marathi
    test_marathi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=/home/krish/content/Version3/wmt22/marathi_test.txt --outputs=/home/krish/content/Version3/wmt22_spm/test.spm.mr'

    # Command to test Hindi
    test_hindi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=/home/krish/content/Version3/wmt22/hindi_test.txt --outputs=/home/krish/content/Version3/wmt22_spm/test.spm.hi'

    # Execute commands
    os.system(marathi_command)
    os.system(hindi_command)
    os.system(dev_marathi_command)
    os.system(dev_hindi_command)
    os.system(test_marathi_command)
    os.system(test_hindi_command)

except Exception as e:
    print(f"Error: {e}")
