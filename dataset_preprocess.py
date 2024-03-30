import os
import pandas as pd
import collections
import json
import os
from datasets import Dataset,DatasetDict
from dataset_prep.info_table_json import load_tables,process_foreign_keys
from dataset_prep.info_db import get_tables
from dataset_prep.get_schema import get_sql_for_database
from static_info import db_schema,db_table_information

class MimicDataset:
    base_data_dir = 'data/mimic_iv'
    db_id = 'mimic_iv'
    # File paths for the dataset and labels
    tables_path = os.path.join(base_data_dir, 'tables.json')               # JSON containing database schema
    train_data_path = os.path.join(base_data_dir, 'train', 'data.json')    # JSON file with natural language questions for training data
    train_label_path = os.path.join(base_data_dir, 'train', 'label.json')  # JSON file with corresponding SQL queries for training data
    valid_data_path = os.path.join(base_data_dir, 'valid', 'data.json')    # JSON file for validation data
    db_path = os.path.join(base_data_dir, f'{db_id}.sqlite')
    test_data_path = os.path.join(base_data_dir, 'test', 'data.json')
    
    def process_data(self,schema_type="static",table_details="table",skip_null=False): #table details= table.json / db
        if schema_type =='static':
            schema=db_schema.strip()
        elif schema_type =='sql':
            schema = get_sql_for_database(self.db_path)
        if table_details =='table':
            table_info = load_tables(self.tables_path)
            foreign_key = process_foreign_keys(table_info[self.db_id])
        elif table_details == 'sql':
            table_info = get_tables(self.db_path) 
            # need to work here
        final_schema = f"{schema}\n{foreign_key}"
        
        with open(self.train_data_path) as f:
            train_data = json.load(f)
        with open(self.train_label_path) as f:
            label_data = json.load(f)
    
        train_data_full = train_data["data"]
        new_data=[]
        for idx , item in enumerate(train_data_full):
            res={}
            id_ = item["id"]
            res["id"]=id_
            res["question"]=item["question"]
            if id_ in label_data:
                label_ = label_data[id_]
                if label_ =="null":
                    if skip_null:
                        continue
                    res["label"]="I do not know"
                    res["numerical_label"] = 0
                else:
                    res["label"]=label_
                    res["numerical_label"] = 1
            else:
                print("not availabel",id_)
                res["label"]="I do not know"
                res["numerical_label"] = 1
            res["schema"] = final_schema
            res["table_description"] = db_table_information.strip()
            
            new_data.append(res)
        
        
        return Dataset.from_list(new_data)
        
    def process_val_data(self):
        # now static , table
        schema=db_schema.strip()
        table_info = load_tables(self.tables_path)
        foreign_key = process_foreign_keys(table_info[self.db_id])
        final_schema = f"{schema}\n{foreign_key}"
        
        with open(self.valid_data_path) as f:
            val_data = json.load(f)  
        val_data_full = val_data["data"]
        for idx , item in enumerate(val_data_full):
            item["schema"] = final_schema
            item["table_description"] = db_table_information.strip()
        
        return Dataset.from_list(val_data_full)
    
    
    def process_test_data(self):
        # now static , table
        schema=db_schema.strip()
        table_info = load_tables(self.tables_path)
        foreign_key = process_foreign_keys(table_info[self.db_id])
        final_schema = f"{schema}\n{foreign_key}"
        
        with open(self.test_data_path) as f:
            val_data = json.load(f)
            
        val_data_full = val_data["data"]
        new_data=[]
        for idx , item in enumerate(val_data_full):
            res={}
            id_ = item["id"]
            res["id"]=id_
            res["question"]=item["question"]
            
            res["schema"] = final_schema
            res["table_description"] = db_table_information.strip()
            new_data.append(res)
        return Dataset.from_list(new_data)
    
    
    