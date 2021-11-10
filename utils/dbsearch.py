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


#=================================================================
# Pre-defined SQL Queries of DB
# Domain: position
# Slots as search conditions: position_name, operation
# Tables: position
#=================================================================
def query_position_name(db_file, position_name):
    conn = create_connection(db_file)
    if conn != None:
        cur = conn.cursor()
        sql_serch_name = "SELECT * FROM position WHERE position_name='" + position_name + "'"
        cur.execute(sql_serch_name)
        rows = cur.fetchall()
        if (len(rows) > 0):
            return "detected"

    return "undetected"

def query_position(db_file, position_name, operation):
    conn = create_connection(db_file)
    if conn != None:
        cur = conn.cursor()
        if (operation == 'add'):
            # search if the name is exist
            sql_serch_name = "SELECT * FROM position WHERE position_name='" + position_name + "'"
            cur.execute(sql_serch_name)
            rows = cur.fetchall()
            if (len(rows)>0):
                return "detected"
            else:
                sql_add_position = "INSERT INTO position(position_name) VALUES('" + position_name + "')"
                cur.execute(sql_add_position)
                conn.commit()
                return "undetected"
        elif (operation == 'move'):
            # search if the name is exist
            sql_serch_name = "SELECT * FROM position WHERE position_name='" + position_name + "'"
            cur.execute(sql_serch_name)
            rows = cur.fetchall()
            if (len(rows) > 0):
                return "detected"
            else:
                return "undetected"

    return "unknown"
#=================================================================
# Pre-defined SQL Queries of DB
# Domain: relocation
# Slots as search conditions: object
# Tables: object
#=================================================================
def query_object(db_file, object_name):
    print(object_name)
    conn = create_connection(db_file)
    if conn != None:
        cur = conn.cursor()
        sql_query = "SELECT * FROM object WHERE object_name='" + object_name + "'"
        cur.execute(sql_query)
        rows = cur.fetchall()
        num_of_rows = len(rows)
        if num_of_rows > 0:
            print(num_of_rows)
            return "detected"

    return "undetected"


#=================================================================
# extract the belief state and searching database
# Return the searching results from database
#=================================================================
def db_search(cfg, context_pred_belief):

    loc_db_req = context_pred_belief.find('<|DB_req|>')
    loc_db_opt = context_pred_belief.find('<|DB_opt|>')
    loc_t_req = context_pred_belief.find('<|T_req|>')
    loc_t_opt = context_pred_belief.find('<|T_opt|>')
    loc_eob = context_pred_belief.find('<|eob|>')
    # get db_req slots
    if loc_db_req > 0:
        if loc_db_opt > 0:
            db_req = context_pred_belief[loc_db_req + 10:loc_db_opt]
        elif loc_t_req > 0:
            db_req = context_pred_belief[loc_db_req + 10:loc_t_req]
        elif loc_t_opt > 0:
            db_req = context_pred_belief[loc_db_req + 10:loc_t_opt]
        else:
            db_req = context_pred_belief[loc_db_req + 10:loc_eob]

        db_req = [x for x in list(dict.fromkeys(db_req.split(' '))) if x]

    # get t_req slots
    if loc_t_req > 0:
        if loc_t_opt > 0:
            t_req = context_pred_belief[loc_t_req + 9:loc_t_opt]
        else:
            t_req = context_pred_belief[loc_t_req + 9:loc_eob]

        t_req = [x for x in list(dict.fromkeys(t_req.split(' '))) if x]

    results = ''
    # db_req slots
    if loc_db_req > 0:
        # get the domain
        domain = db_req[0]
        if domain == 'delivery':
            # set the condition for searching area_location table
            area, location = '', ''
            for item in db_req[1:]:
                key, value = item[0:item.find('=')], item[item.find('=') + 1:]
                if key == 'area':
                    area = value
                elif key == 'location':
                    location = value
            if (area == 'not_mentioned') and (location == 'not_mentioned'):
                results = 'area=null location=null'
            elif (area != 'not_mentioned') and (location == 'not_mentioned'):
                db_results = query_area(cfg.dataset_path_production_db, area)
                results = 'area=' + db_results + ' location=null'
            elif (area != 'not_mentioned') and (location != 'not_mentioned'):
                db_results = query_area_location(cfg.dataset_path_production_db, area, location)
                results = 'area=' + db_results + ' location=' + db_results
            elif (area == 'not_mentioned') and (location != 'not_mentioned'):
                db_results = query_location(cfg.dataset_path_production_db, location)
                results = 'area=null' + ' location=' + db_results

        elif domain == 'assembly':
            # set the condition for searching product table
            producttype = ''
            for item in db_req[1:]:
                key, value = item[0:item.find('=')], item[item.find('=') + 1:]
                if key == 'producttype':
                    producttype = value
            if (producttype == 'not_mentioned'):
                results = 'producttype=null'
            else:
                db_results, _ = query_product(cfg.dataset_path_production_db, producttype)
                results = 'producttype=' + db_results

        elif domain == 'position':
            # set the condition for searching position table
            position_name = ''
            for item in db_req[1:]:
                key, value = item[0:item.find('=')], item[item.find('=') + 1:]
                if key == 'position_name':
                    producttype = value
            if (position_name == 'not_mentioned'):
                results = 'position_name=null'
            else:
                db_results = query_position_name(cfg.dataset_path_production_db, position_name)
                results = 'position_name=' + db_results

        elif domain == 'relocation':
            # set the condition for searching position table
            object_name = ''
            for item in db_req[1:]:
                key, value = item[0:item.find('=')], item[item.find('=') + 1:]
                if key == 'object_name':
                    object_name = value
            if (object_name == 'not_mentioned'):
                results = 'object_name=null'
            else:
                print(object_name)
                db_results = query_object(cfg.dataset_path_production_db, object_name)
                results = 'object_name=' + db_results

    # t_req slots
    if loc_t_req > 0:
        # get the domain
        domain = t_req[0]
        if domain == 'delivery':
            # set the condition for searching area_location table
            object = ''
            for item in t_req[1:]:
                key, value = item[0:item.find('=')], item[item.find('=') + 1:]
                if key == 'object':
                    object = value
            if (object == 'not_mentioned'):
                results += ' object=null'
            else:
                results += ' object=detected'

        elif domain == 'assembly':
            # set the condition for searching product table
            product, quantity = '', ''
            for item in t_req[1:]:
                key, value = item[0:item.find('=')], item[item.find('=') + 1:]
                if key == 'product':
                    product = value
                elif key == 'quantity':
                    quantity = value
            if (product == 'not_mentioned'):
                results += ' product=null'
            else:
                results += ' product=detected'
            if (quantity == 'not_mentioned'):
                results += ' quantity=null'
            else:
                results += ' quantity=detected'

        elif domain == 'position':
            operation = ''
            for item in t_req[1:]:
                key, value = item[0:item.find('=')], item[item.find('=') + 1:]
                if key == 'operation':
                    operation = value
            if (operation == 'not_mentioned'):
                results += ' operation=null'
            else:
                results += ' operation=detected'

    return results
