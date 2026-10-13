# import datetime

# launch_date = datetime.date(day=10, month=2, year=2022)

# some_future = launch_date + datetime.timedelta(days=42)

# print(some_future)

from datetime import datetime, timedelta
times= datetime(month=3, day=8, year=2022) + timedelta(days=3, hours=5, minutes=6)

print(times)