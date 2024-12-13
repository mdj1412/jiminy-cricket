import os

from nltk.tokenize import wordpunct_tokenize

from datasets import Dataset

# prepare the dataset
path = "/MIR_NAS/jerry0110_1/madrag/madrag/processed_data/"

text_anomal_image_normal_path = os.path.join(path, "text_anomal_image_normal")
text_normal_image_anomal_path = os.path.join(path, "text_normal_image_anomal")

text_an_image_n_file = os.path.join(text_anomal_image_normal_path, "data-00000-of-00001.arrow")
text_n_image_an_file = os.path.join(text_normal_image_anomal_path, "data-00000-of-00001.arrow")

# Load the .arrow file into a Dataset
an_n_dataset = Dataset.from_file(text_an_image_n_file)
n_an_dataset = Dataset.from_file(text_n_image_an_file)

an_n_token_length, n_an_token_length = [], []

# MUN-lang : text_anomal_image_normal
for data in an_n_dataset:
    caption = data['caption']

    # wordpunct_tokenize를 사용하여 토큰화
    tokens = wordpunct_tokenize(caption)
    an_n_token_length.append(len(tokens))

sorted_list = an_n_token_length # deep copy
n = len(sorted_list)
median = sorted_list[n//2] if n%2==1 else (sorted_list[n//2-1] + sorted_list[n//2]) / 2
print (sum(an_n_token_length), sum(an_n_token_length)/n, median)


# MUN-vis : text_normal_image_anomal
for data in n_an_dataset:
    caption = data['caption']

    # wordpunct_tokenize를 사용하여 토큰화
    tokens = wordpunct_tokenize(caption)
    n_an_token_length.append(len(tokens))

sorted_list = n_an_token_length # deep copy
n = len(sorted_list)
median = sorted_list[n//2] if n%2==1 else (sorted_list[n//2-1] + sorted_list[n//2]) / 2
print (sum(n_an_token_length), sum(n_an_token_length)/n, median)

"""
MUN-lang
sum: 4742, avg: 9.427435387673956 median: 7

MUN-vis
sum: 5940, avg: 5.5722326454033775, median: 6.0
"""

from IPython import embed; embed()