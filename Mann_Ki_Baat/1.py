import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the Excel file directly from the server
train_df = pd.read_excel('/home/krish/content/Mann_Ki_Baat/train_data(V5).xlsx')
test_df =pd.read_excel('/home/krish/content/Mann_Ki_Baat/farmer.xlsx')
dev_df=pd.read_excel('/home/krish/content/Mann_Ki_Baat/validation_data.xlsx')

# Create the directory for saving files
os.makedirs("/home/krish/content/Mann_Ki_Baat/wmt22", exist_ok=True)

# Save Mundari sentences to text files on the remote server
train_df['MUNDARI'].to_csv('/home/krish/content/Mann_Ki_Baat/wmt22/marathi_train.txt', index=False, header=False)
test_df['MUNDARI'].to_csv('/home/krish/content/Mann_Ki_Baat/wmt22/marathi_test.txt', index=False, header=False)
dev_df['MUNDARI'].to_csv('/home/krish/content/Mann_Ki_Baat/wmt22/marathi_dev.txt', index=False, header=False)

# Save Hindi sentences to text files on the remote server
train_df['HINDI'].to_csv('/home/krish/content/Mann_Ki_Baat/wmt22/hindi_train.txt', index=False, header=False)
test_df['HINDI'].to_csv('/home/krish/content/Mann_Ki_Baat/wmt22/hindi_test.txt', index=False, header=False)
dev_df['HINDI'].to_csv('/home/krish/content/Mann_Ki_Baat/wmt22/hindi_dev.txt', index=False, header=False)
