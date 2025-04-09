def merge_dictionaries(hi_dict_path, mr_dict_path, output_path):
    # Read Hindi dictionary
    with open(hi_dict_path, "r", encoding="utf-8") as hi_file:
        hi_vocab = set(hi_file.readlines())

    # Read Marathi dictionary
    with open(mr_dict_path, "r", encoding="utf-8") as mr_file:
        mr_vocab = set(mr_file.readlines())

    # Merge vocabularies
    shared_vocab = sorted(hi_vocab.union(mr_vocab))

    # Write to output file
    with open(output_path, "w", encoding="utf-8") as shared_file:
        shared_file.writelines(shared_vocab)

# Paths to input dictionaries and output shared dictionary
hi_dict_path = "/home/krish/content/trial/wmt22_spm/wmt22_bin/dict.hi.txt"
mr_dict_path = "/home/krish/content/trial/wmt22_spm/wmt22_bin/dict.mr.txt"
output_path = "/home/krish/content/trial/wmt22_spm/wmt22_bin/shared.dict.txt"

merge_dictionaries(hi_dict_path, mr_dict_path, output_path)
print(f"Shared dictionary created at: {output_path}")
