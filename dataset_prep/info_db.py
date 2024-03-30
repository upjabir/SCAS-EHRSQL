import collections
import json
import os
import re
import sqlite3

class SqliteTable(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def get_tables(path_db):
    if not os.path.exists(path_db):
        raise RuntimeError(f"{path_db} not exists")

    # init sqlite connection
    connection = sqlite3.connect(path_db)
    cur = connection.cursor()

    # extract table information
    table_info = parse_db(path_db, cur)
    print("table info",table_info)
    # TODO: ! add here
    table_names = get_table_names(cur=cur)

    res = list()
    for table_name in table_names:
        # schema
        schema = [_[1] for _ in cur.execute(f"PRAGMA table_info({table_name})")]
        print("schema",schema)
        # data
        data = None
        # data = cur.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchall()

        # append table
        res.append(
            SqliteTable(
                name=table_name,
                schema=schema,
                data=data,
                table_info=table_info.get(table_name, dict())
            )
        )

    cur.close()
    return res


def parse_db(path_db, cur=None):
    """Parse the sql file and extract primary and foreign keys

    :param path_file:
    :return:
    """
    table_info = dict()
    table_names = get_table_names(path_db, cur)
    print("table name",table_names)
    for table_name in table_names:
        pks = get_primary_key(table_name, path_db,cur)
        fks = get_foreign_key(table_name, path_db, cur)

        table_info[table_name] = {
            "primary_key": pks,
            "foreign_key": fks
        }
    return table_info


def execute_query(queries, path_db=None, cur=None):
    """Execute queries and return results. Reuse cur if it's not None.

    """
    assert not (path_db is None and cur is None), "path_db and cur cannot be NoneType at the same time"

    close_in_func = False
    if cur is None:
        con = sqlite3.connect(path_db)
        cur = con.cursor()
        close_in_func = True

    if isinstance(queries, str):
        results = cur.execute(queries).fetchall()
    elif isinstance(queries, list):
        results = list()
        for query in queries:
            res = cur.execute(query).fetchall()
            results.append(res)
    else:
        raise TypeError(f"queries cannot be {type(queries)}")

    # close the connection if needed
    if close_in_func:
        con.close()

    return results

def format_foreign_key(table_name: str, res: list):
    # FROM: self key | TO: target key
    res_clean = list()
    for row in res:
        table, source, to = row[2:5]
        row_clean = f"({table_name}.{source}, {table}.{to})"
        res_clean.append(row_clean)
    return res_clean


def get_foreign_key(table_name, path_db=None, cur=None):
    res_raw = execute_query(f"PRAGMA foreign_key_list({table_name})", path_db, cur)
    res = format_foreign_key(table_name, res_raw)
    print(res_raw)
    return res


def get_primary_key(table_name, path_db=None, cur=None):
    res_raw = execute_query(f'PRAGMA table_info({table_name})', path_db, cur)
    pks = list()
    for row in res_raw:
        if row[5] == 1:
            pks.append(row[1])
    print(pks)
    return pks


def get_table_names(path_db=None, cur=None):
    """Get names of all tables within the database, and reuse cur if it's not None

    """
    table_names = execute_query(queries="SELECT name FROM sqlite_master WHERE type='table'", path_db=path_db, cur=cur)
    table_names = [_[0] for _ in table_names]
    return table_names