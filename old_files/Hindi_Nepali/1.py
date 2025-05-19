import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Replace with the URL or path to your Excel file on the server
excel_file_url = '/home/krish/content/Merged_Data.xlsx'

# Load the Excel file directly from the server
df = pd.read_excel(excel_file_url)

# Split the data into train, test, and dev sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)  # 80% train, 20% temp
test_df, dev_df = train_test_split(temp_df, test_size=0.5, random_state=42)  # 10% test, 10% dev

# Create the directory for saving files
os.makedirs("/home/krish/content/Hindi_Nepali/wmt22", exist_ok=True)

# Save Mundari sentences to text files on the remote server
train_df['MUNDARI'].to_csv('/home/krish/content/Hindi_Nepali/wmt22/nepali_train.txt', index=False, header=False)
test_df['MUNDARI'].to_csv('/home/krish/content/Hindi_Nepali/wmt22/nepali_test.txt', index=False, header=False)
dev_df['MUNDARI'].to_csv('/home/krish/content/Hindi_Nepali/wmt22/nepali_dev.txt', index=False, header=False)

# Save Hindi sentences to text files on the remote server
train_df['HINDI'].to_csv('/home/krish/content/Hindi_Nepali/wmt22/hindi_train.txt', index=False, header=False)
test_df['HINDI'].to_csv('/home/krish/content/Hindi_Nepali/wmt22/hindi_test.txt', index=False, header=False)
dev_df['HINDI'].to_csv('/home/krish/content/Hindi_Nepali/wmt22/hindi_dev.txt', index=False, header=False)
