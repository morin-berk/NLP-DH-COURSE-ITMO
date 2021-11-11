import pandas as pd
import random

tsv_file = 'reviews.tsv'
csv_table = pd.read_table(tsv_file, sep='\t')
csv_table.to_csv('new_name.csv', index=False, encoding="utf8")

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
