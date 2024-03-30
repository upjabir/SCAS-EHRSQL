from argparse import ArgumentParser
import json
import os

def get_args():
    parser = ArgumentParser(description="LLM SELECTOR")
    parser.add_argument("--sql_generated_data_path", type=str)
    parser.add_argument("--classification_data_path", type=str)
    parser.add_argument("--final_result_path", type=str)
    arguments = parser.parse_args()
    return arguments

def write_json(path, file):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, 'w+') as f:
        json.dump(file, f)
        
def main(args):
    with open(args.classification_data_path) as f:
        cls_data=json.load(f)

    with open(args.sql_generated_data_path) as f:
        sql_data = json.load(f)
        
    for key,value in sql_data.items():
        cls_value = cls_data[key]
        if cls_value ==0:
            value="null"
        sql_data[key] = value
    write_json(args.final_result_path,sql_data)
    
if __name__ == "__main__":
    args = get_args()
    main(args)