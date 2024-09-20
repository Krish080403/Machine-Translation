import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the Excel file directly from the server
train_df = pd.read_excel('/home/krish/content/Version2/train_data(V2).xlsx')
test_df =pd.read_excel('/home/krish/content/Version2/test_data.xlsx')
dev_df=pd.read_excel('/home/krish/content/Version2/validation_data.xlsx')

# Create the directory for saving files
os.makedirs("/home/krish/content/Version2/wmt22", exist_ok=True)

# Save Mundari sentences to text files on the remote server
train_df['MUNDARI'].to_csv('/home/krish/content/Version2/wmt22/marathi_train.txt', index=False, header=False)
test_df['MUNDARI'].to_csv('/home/krish/content/Version2/wmt22/marathi_test.txt', index=False, header=False)
dev_df['MUNDARI'].to_csv('/home/krish/content/Version2/wmt22/marathi_dev.txt', index=False, header=False)

# Save Hindi sentences to text files on the remote server
train_df['HINDI'].to_csv('/home/krish/content/Version2/wmt22/hindi_train.txt', index=False, header=False)
test_df['HINDI'].to_csv('/home/krish/content/Version2/wmt22/hindi_test.txt', index=False, header=False)
dev_df['HINDI'].to_csv('/home/krish/content/Version2/wmt22/hindi_dev.txt', index=False, header=False)
