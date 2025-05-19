import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Replace with the URL or path to your Excel file on the server
excel_file_url = '/home/krish/content/merged.xlsx'

# Load the Excel file directly from the server
df = pd.read_excel(excel_file_url)

test_df=df

# Create the directory for saving files
os.makedirs("/home/krish/content/Hindi_Marathi/report/wmt22", exist_ok=True)

test_df['MUNDARI'].to_csv('/home/krish/content/Hindi_Marathi/report/wmt22/marathi_test.txt', index=False, header=False)
test_df['HINDI'].to_csv('/home/krish/content/Hindi_Marathi/report/wmt22/hindi_test.txt', index=False, header=False)
