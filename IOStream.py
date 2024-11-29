import os
import re
from texttable import Texttable


class IOStream():
    def __init__(self, path):
        self.file = open(path, 'a')

    def cprint(self, text):
        self.file.write(text + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


def table_printer(args):
    args = vars(args)
    keys = sorted(args.keys()) 
    table = Texttable()
    table.set_cols_dtype(['t', 't']) 
    rows = [["Parameter", "Value"]] 
    for k in keys:
        rows.append([k.replace("_", " ").capitalize(), str(args[k])]) 
    table.add_rows(rows)
    return table.draw()


def exp_init(exp_name, formatted_time):
    path = os.path.join('outputs', formatted_time, exp_name)
    os.makedirs(path, exist_ok=True)
    return path
