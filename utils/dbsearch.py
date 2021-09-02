import sqlite3
from sqlite3 import Error
from config import Config

#=================================================================
# Public functions for DB query
#=================================================================
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


#=================================================================
# Pre-defined SQL Queries of DB
# Domain: delivery
# Slots as search conditions: area, location, sender, recipient
# Tables: area_location, employee
#=================================================================
# query area
def query_area(db_file, area):

    conn = create_connection(db_file)
    if conn != None:
        cur = conn.cursor()
        sql_query = "SELECT * FROM area_location WHERE area='" + area + "'"
        cur.execute(sql_query)
        rows = cur.fetchall()
        num_of_rows = len(rows)
        if num_of_rows > 0:
            print(num_of_rows)
            return "detected"

    return "undetected"

# query location
def query_location(db_file, location):

    conn = create_connection(db_file)
    if conn != None:
        cur = conn.cursor()
        sql_query = "SELECT * FROM area_location WHERE location='" + location + "'"
        cur.execute(sql_query)
        rows = cur.fetchall()
        num_of_rows = len(rows)
        if num_of_rows > 0:
            print(num_of_rows)
            return "detected"

    return "undetected"


# query area and location
def query_area_location(db_file, area, location):
    conn = create_connection(db_file)
    if conn != None:
        cur = conn.cursor()
        sql_query = "SELECT * FROM area_location WHERE area='" + area + "' and location='" + location + "'"
        cur.execute(sql_query)
        rows = cur.fetchall()
        num_of_rows = len(rows)
        if num_of_rows > 0:
            print(num_of_rows)
            return "detected"

    return "undetected"


# query sender/recipient
def query_worker_name(db_file, worker_name):
    conn = create_connection(db_file)
    if conn != None:
        cur = conn.cursor()
        sql_query = "SELECT * FROM employee WHERE name='" + worker_name + "'"
        cur.execute(sql_query)
        rows = cur.fetchall()
        num_of_rows = len(rows)
        if num_of_rows > 0:
            print(num_of_rows)
            return "detected"

    return "undetected"


#=================================================================
# Pre-defined SQL Queries of DB
# Domain: assembly
# Slots as search conditions: product
# Tables: area_location, employee
#=================================================================
# query area
def query_product(db_file, product):
    conn = create_connection(db_file)
    if conn != None:
        cur = conn.cursor()
        sql_query = "SELECT * FROM product WHERE product='" + product + "'"
        cur.execute(sql_query)
        rows = cur.fetchall()
        num_of_rows = len(rows)
        if num_of_rows > 0:
            for row in rows:
                return "detected", row[2]

    return "undetected", "no details"

# for testing
if __name__ == "__main__":
    cfg = Config()
    result, details = query_product(cfg.dataset_path_production_db, "phone")
    print(result, details)
