import sqlite3
def get_table_names(path_db=None, cur=None):
    """Get names of all tables within the database, and reuse cur if it's not None

    """
    table_names = execute_query(queries="SELECT name FROM sqlite_master WHERE type='table'", path_db=path_db, cur=cur)
    table_names = [_[0] for _ in table_names]
    return table_names

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
    

def get_sql_for_database(path_db=None, cur=None):
    close_in_func = False
    if cur is None:
        con = sqlite3.connect(path_db)
        cur = con.cursor()
        close_in_func = True

    table_names = get_table_names(path_db, cur)

    queries = [f"SELECT sql FROM sqlite_master WHERE tbl_name='{name}'" for name in table_names]

    sqls = execute_query(queries, path_db, cur)

    if close_in_func:
        cur.close()

    return [_[0][0] for _ in sqls]