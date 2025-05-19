import os

try:
    # Command to train Nepali
    nepali_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=/home/krish/content/Hindi_Nepali/wmt22/nepali_train.txt --outputs=/home/krish/content/Hindi_Nepali/wmt22_spm/train.spm.ne'
    
    # Command to train Hindi
    hindi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=/home/krish/content/Hindi_Nepali/wmt22/hindi_train.txt --outputs=/home/krish/content/Hindi_Nepali/wmt22_spm/train.spm.hi'

    # Command to validate Nepali
    dev_nepali_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=/home/krish/content/Hindi_Nepali/wmt22/nepali_dev.txt --outputs=/home/krish/content/Hindi_Nepali/wmt22_spm/dev.spm.ne'

    # Command to validate Hindi
    dev_hindi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=/home/krish/content/Hindi_Nepali/wmt22/hindi_dev.txt --outputs=/home/krish/content/Hindi_Nepali/wmt22_spm/dev.spm.hi'

    # Command to test Nepali
    test_nepali_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=/home/krish/content/Hindi_Nepali/wmt22/nepali_test.txt --outputs=/home/krish/content/Hindi_Nepali/wmt22_spm/test.spm.ne'

    # Command to test Hindi
    test_hindi_command = 'python3.10 /home/krish/content/spm_encode.py --model /home/krish/content/spm.128k.model --output_format=piece --inputs=/home/krish/content/Hindi_Nepali/wmt22/hindi_test.txt --outputs=/home/krish/content/Hindi_Nepali/wmt22_spm/test.spm.hi'

    # Execute commands
    os.system(nepali_command)
    os.system(hindi_command)
    os.system(dev_nepali_command)
    os.system(dev_hindi_command)
    os.system(test_nepali_command)
    os.system(test_hindi_command)

except Exception as e:
    print(f"Error: {e}")
