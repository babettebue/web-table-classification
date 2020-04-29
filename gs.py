import sqlite3
import pandas as pd
import os

def load_gs(path):
    
    #Create connection to SQLite mannheim.db
    conn = sqlite3.connect(path)
    #Get table
    cur = conn.cursor()
    cur.execute("SELECT * FROM 'table'")
    rows = cur.fetchall()
    df =pd.DataFrame(rows)
    df.columns = ['id', 'pageTitle', 'url', 'title', 'cells', 'label', 'recordEndOffset', 'recordOffset', 's3Link', 'tableNum', 'htmlCode']
    
    conn.close()
    
    return df