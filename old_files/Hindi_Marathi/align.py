def align_sentences(hindi_file, marathi_file):
    """
    Align sentences between Hindi and Marathi files and overwrite the original files with aligned sentences.
    """
    # Read Hindi sentences
    with open(hindi_file, 'r', encoding='utf-8') as hi_f:
        hindi_sentences = hi_f.readlines()

    # Read Marathi sentences
    with open(marathi_file, 'r', encoding='utf-8') as mr_f:
        marathi_sentences = mr_f.readlines()

    # Check if the number of sentences matches
    if len(hindi_sentences) != len(marathi_sentences):
        print(f"Number of sentences mismatch: {len(hindi_sentences)} Hindi vs {len(marathi_sentences)} Marathi")
        print("Aligning the sentences...")

        # Align sentences by keeping only the ones where both files have a corresponding sentence
        aligned_hindi_sentences = []
        aligned_marathi_sentences = []

        for hi_sentence, mr_sentence in zip(hindi_sentences, marathi_sentences):
            if hi_sentence.strip() and mr_sentence.strip():
                aligned_hindi_sentences.append(hi_sentence.strip())
                aligned_marathi_sentences.append(mr_sentence.strip())

        # Overwrite the original files with aligned sentences
        with open(hindi_file, 'w', encoding='utf-8') as hi_f:
            hi_f.write("\n".join(aligned_hindi_sentences) + "\n")

        with open(marathi_file, 'w', encoding='utf-8') as mr_f:
            mr_f.write("\n".join(aligned_marathi_sentences) + "\n")

        print(f"Aligned sentences saved to {hindi_file} and {marathi_file}")
    else:
        print(f"Number of sentences match for {hindi_file} and {marathi_file}. No alignment needed.")

def align_all_datasets():
    """
    Align sentences for train, dev, and test datasets.
    """

    # Paths for Train dataset
    align_sentences(
        hindi_file='/home/krish/content/Hindi_Marathi/wmt22_spm/train.spm.hi',
        marathi_file='/home/krish/content/Hindi_Marathi/wmt22_spm/train.spm.mr'
    )

    # Paths for Dev dataset
    align_sentences(
        hindi_file='/home/krish/content/Hindi_Marathi/wmt22_spm/dev.spm.hi',
        marathi_file='/home/krish/content/Hindi_Marathi/wmt22_spm/dev.spm.mr'
    )

    # Paths for Test dataset
    align_sentences(
        hindi_file='/home/krish/content/Hindi_Marathi/wmt22_spm/test.spm.hi',
        marathi_file='/home/krish/content/Hindi_Marathi/wmt22_spm/test.spm.mr'
    )

if __name__ == "__main__":
    align_all_datasets()
