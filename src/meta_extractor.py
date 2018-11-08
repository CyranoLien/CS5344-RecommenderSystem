import json
from tqdm import tqdm

INP_PATH = '../bin/meta_Apps_for_Android.json'
OUT_PATH = '../bin/only_p_all.json'

with open(INP_PATH, 'r') as f1:
    with open(OUT_PATH, 'w') as f2:
        for l in tqdm(f1.readlines()):
            data = eval(l)
            extract = {}
            extract['item'] = [data['asin']]
            if 'related' in data:
                if 'also_bought' in data['related']:
                    extract['also_bought'] = data['related']['also_bought']
                    f2.write(json.dumps(extract))
                    f2.write('\n')
                else:
                    pass
            else:
                pass
        print('Preprocess is finished.')