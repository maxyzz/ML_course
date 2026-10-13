# import os
# import psycopg2
# from dotenv import load_dotenv

# load_dotenv()

# connection = psycopg2.connect(
#     database="startml",
#     user="robot-startml-ro",
#     password=os.environ["password"],
#     host="postgres.lab.karpov.courses",
#     port=6432
# )
# cursor = connection.cursor()

# cursor.close() 
# connection.close()


import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

con = (
    f"postgresql://robot-startml-ro:{os.environ['password']}@"
    "postgres.lab.karpov.courses:6432/startml"
)

df = pd.read_sql(
    """SELECT * FROM "feed_action" LIMIT 10 """,
    con=con,
)

df.head()