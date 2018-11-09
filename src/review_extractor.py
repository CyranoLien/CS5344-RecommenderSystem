import json
from tqdm import tqdm


INP_PATH = '../bin/Apps_for_Android_5.json'
OUT_PATH = '../bin/c&p_data_all.json'

key_map = {'reviewerID': 'customerID', 'asin': 'productID', 'overall': 'score'}


with open(INP_PATH, 'r') as f1:
    with open(OUT_PATH, 'w') as f2:
        for l in tqdm(f1.readlines()):
            data = eval(l)
            extract = lambda data: dict((key_map[i], data[i]) for i in key_map)
            temp = json.dumps(extract(data))
            f2.write(temp)
            f2.write('\n')