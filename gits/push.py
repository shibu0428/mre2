import os
from datetime import datetime

# Commit comment
comment = input("コメント入力")
comment='"'+comment+'"'

#push
os.system('cd ..')
os.system('git add .')
os.system('git commit -m '+comment)
os.system('git push')