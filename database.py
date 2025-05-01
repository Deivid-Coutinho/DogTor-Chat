import sqlite3
from datetime import datetime
import pandas as pd
import os

DB_PATH = "logs/dogtor.db"

def init_db():
    os.makedirs("logs", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resultados (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT NOT NULL,
            resultado TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def salvar_resultado(resultado):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO resultados (data, resultado) VALUES (?, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), resultado))
    conn.commit()
    conn.close()

def carregar_estatisticas():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM resultados", conn)
    conn.close()
    return df
