import pandas as pd
import random

tsv_file = 'reviews.tsv'
csv_table = pd.read_table(tsv_file, sep='\t')
csv_table.to_csv('new_name.csv', index=False, encoding="utf8")

# with open('new_name.csv', 'r', encoding="utf8") as csvf:
#     linecount = sum(1 for line in csvf if line.strip() != '')
#
# # Create index sets for training and test data
# indices = list(range(linecount))
# random.shuffle(indices)
# ind_test = set(indices[:int(linecount*0.2)])
# del indices
#
# # Partition CSV file
# with open('new_name.csv', 'r', encoding="utf8") as csvf, \
#         open('train.csv','w', encoding="utf8") as trainf, \
#         open('test.csv','w', encoding="utf8") as testf:
#     i = 0
#     for line in csvf:
#         if line.strip() != '':
#             if i in ind_test:
#                 testf.write(line.strip() + '\n')
#             else:
#                 trainf.write(line.strip() + '\n')

with open('new_name.csv', encoding="utf8") as data:
    with open('test.csv', 'w', encoding="utf8") as test:
        with open('train.csv', 'w', encoding="utf8") as train:
            header = next(data)
            test.write(header)
            train.write(header)
            for line in data:
                if random.random() > 0.8:
                    train.write(line)
                else:
                    test.write(line)
