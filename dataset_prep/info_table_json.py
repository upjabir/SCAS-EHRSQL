import pandas as pd


def find_foreign_keys_MYSQL_like(spider_foreign,db_name):
  df = spider_foreign[spider_foreign['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + " = " + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ','
  output= output[:-1] + "]"
  return output
def find_fields_MYSQL_like(spider_schema,db_name):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    output += "Table " +name+ ', columns = ['
    for index, row in group.iterrows():
      output += row[" Field Name"]+','
    output = output[:-1]
    output += "]\n"
  return output
def find_primary_keys_MYSQL_like(spider_primary,db_name):
  df = spider_primary[spider_primary['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['Table Name'] + '.' + row['Primary Key'] +','
  output = output[:-1]
  output += "]\n"
  return output
def create_schema(table_json_path):
    schema_df = pd.read_json(table_json_path)  # table.json read
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    spider_schema = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return spider_schema,spider_primary,spider_foreign

def load_tables(path):
    table_data={}
    spider_schema,spider_primary,spider_foreign = create_schema(path)
    db_ids = spider_schema['Database name'].unique()
    for db_id in db_ids:
        data={}
        table_names_columns = find_fields_MYSQL_like(spider_schema,db_id)
        foreign_keys = find_foreign_keys_MYSQL_like(spider_foreign,db_id)
        primary_keys = find_primary_keys_MYSQL_like(spider_primary,db_id)
        data["table_names_columns"] = table_names_columns
        data["foreign_keys"] = foreign_keys
        data["primary_keys"] = primary_keys
        table_data[db_id]=data
    return table_data

def process_foreign_keys(schema):
    foreign_key_str=""
    foreign_keys = schema["foreign_keys"]
    foreign_key_split = foreign_keys.split(",")
    for text in foreign_key_split:
        text = text.replace("[", "").replace("]", "")
        text_split = text.split("=")
        foreign_key_str += f"-- {text_split[0]}can be joined with{text_split[1]}\n"
    return foreign_key_str


