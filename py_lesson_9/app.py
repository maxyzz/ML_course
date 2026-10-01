from fastapi import Depends, FastAPI, HTTPException
import datetime
from pydantic import BaseModel 
import psycopg2
from loguru import logger
from psycopg2.extras import RealDictCursor

con = (
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)
def get_db():
    connection = psycopg2.connect(
        con,
        cursor_factory = RealDictCursor
    )
    return connection

app = FastAPI()
# 4 task
# @app.get("/")
# async def root():
#     return "hello, world"

# 5 task
# @app.get("/")
# async def sum(a:int, b:int):
#     return a + b

# 6 task
# @app.get("/sum_date")
# async def sum(current_date: datetime.date , offset: int):
#     a = (current_date + datetime.timedelta(days=offset))
#     return a

# 7-8 task
# class User(BaseModel):
#     name: str
#     surname: str
#     age: int
#     registration_date: datetime.date

# @app.post("/user/validate")
# def validate(user: User):
#     return f"Will add user: {user.name} {user.surname} with age {user.age}"

# 9-10 task
# @app.get("/user/{user_id}")
# async def get_user(user_id: int):
#     connection = psycopg2.connect(
#         con,
#     cursor_factory = RealDictCursor
#     )
#     cursor = connection.cursor()  
#     cursor.execute(f"""select gender, age, city
#     FROM public."user" users WHERE id = {user_id}
#     """)
#     user = cursor.fetchone()
#     logger.info(f"Get user with id {user}")
#     if user is None:
#         raise HTTPException(status_code=404, detail="user not found")
#     return user      

# task 11
# @app.get("/user/{user_id}")
# def my_func(user_id: int, db=Depends(get_db)):
#     with db.cursor() as cursor:
#         cursor.execute(f"""select gender, age, city
#         FROM public."user" users WHERE id = {user_id}
#         """)
#         user = cursor.fetchone()
#     logger.info(f"Get user with id {user}")
#     if user is None:
#         raise HTTPException(status_code=404, detail="user not found")
#     return user

# task 11
class PostResponse(BaseModel):
    id: int
    text: str
    topic: str 

    class Config:
        orm_mode = True

@app.get("/post/{id}",response_model=PostResponse)
def my_func(id: int, db=Depends(get_db)):
    with db.cursor() as cursor:
        cursor.execute(f"""SELECT id, "text", topic
                                FROM public.post 
                                where id= {id}
                        """)
        post = cursor.fetchone()
    logger.info(f"Get post with id {post}")
    if post is None:
        raise HTTPException(status_code=404, detail="post not found")
    return PostResponse(**post)