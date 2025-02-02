import sqlite3

conn = sqlite3.connect('chess_games.db')
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT username FROM games ORDER BY rating DESC LIMIT 200")
for row in cursor.fetchall():
   print(row)
conn.close()