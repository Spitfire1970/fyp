import berserk
import os
from dotenv import load_dotenv
from datetime import datetime

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

load_dotenv()

TOKEN = os.getenv('LICHESS_API_TOKEN')

session = berserk.TokenSession(TOKEN)
client = berserk.Client(session)
account = client.account.get()
print(account)
leader = client.users.get_leaderboard('blitz', count=300)
print(len(leader))
print('------'*10)
# nihal_inf = client.users.get_public_data('nihalsarin2004')
# print(nihal_inf)
# print(type(nihal_inf))
try:
   info = client.users.get_public_data("nihalsarin2004wefwef")
except berserk.exceptions.ResponseError as e:
   if e.status_code == 429: print('hey', e.status_code, type(e.status_code ))
   else:
      print(e, 'hey')
print(nihal_inf['seenAt'])
print(nihal_inf['disabled'])
print(type(nihal_inf['perfs']['bullet']['rating']))
# start = berserk.utils.to_millis(datetime(2018, 12, 8))
# end = berserk.utils.to_millis(datetime(2024, 12, 9))
# print('------'*10)
# ob = list(client.games.export_by_player('nihalsarin2004', since=start, until=end, max = 1))
# print(ob)
# print(len(ob))