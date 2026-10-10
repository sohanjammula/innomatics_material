# from langchain_community.chat_message_histories import SQLChatMessageHistory

# # Inspect the SQL chat database
# history = SQLChatMessageHistory(session_id="123", connection="sqlite:///conversations.db")
# print(history.messages)

import sqlite3

conn = sqlite3.connect("conversations.db")
cursor = conn.cursor()

# cursor.execute("""
# SELECT name
# FROM sqlite_master
# WHERE type='table'
# ORDER BY name;
# """)

# tables = cursor.fetchall()

# print(tables)

cursor.execute("""
select * from message_store 
""")

data = cursor.fetchall()

for each_row in data:
    print(each_row)