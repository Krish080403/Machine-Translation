import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Create output directory
    output_dir = "/home/krish/content/trial/extra_loss"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created directory: {output_dir}")
    
    # Load the Excel files
    logger.info("Loading Excel files...")
    train_df = pd.read_excel('/home/krish/content/trial/train.xlsx')
    dev_df = pd.read_excel('/home/krish/content/trial/validation.xlsx')
    test_df = pd.read_excel('/home/krish/content/trial/test.xlsx')
    
    # Check if 'EXTRA_LOSS' column exists
    # If not, you'll need to create it based on your requirements
    for df, name in [(train_df, 'train'), (dev_df, 'valid'), (test_df, 'test')]:
        if 'EXTRA_LOSS' not in df.columns:
            logger.warning(f"'EXTRA_LOSS' column not found in {name} dataframe.")
            logger.info(f"Creating a placeholder EXTRA_LOSS column for {name} dataframe.")
            # This is a placeholder - replace with your actual loss calculation
            df['EXTRA_LOSS'] = 0.5  # Example placeholder value
    
    # Save loss values to files
    logger.info("Saving loss values to files...")
    train_df['EXTRA_LOSS'].to_csv(f'{output_dir}/train.loss', index=False, header=False)
    dev_df['EXTRA_LOSS'].to_csv(f'{output_dir}/valid.loss', index=False, header=False)
    test_df['EXTRA_LOSS'].to_csv(f'{output_dir}/test.loss', index=False, header=False)
    
    logger.info("Loss values saved successfully.")
    
    # Print dataset statistics
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Dev set: {len(dev_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")

except Exception as e:
    logger.error(f"Error processing Excel files: {e}")