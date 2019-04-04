# %%

import pandas as pd
import glob
import pathlib

# import re
import pickle as pk

# reg_space = re.compile(r"[\w']+|[^\w ]")
# %%
# from_glob = "./sample.txt"
from_glob = "./datasets/test*"

files = [pathlib.Path(f) for f in glob.glob(from_glob)]
data = []
[data.extend(pk.load(f.open("rb"))) for f in files]
# %%
def get_lens(dialogs):
    return [len(sum(i, [])) for i in dialogs if i]


persona, utters = list(zip(*data))
# %%
persona_lens, utters_lens = get_lens(persona), get_lens(utters)
# %%
# df = pd.DataFrame((persona_lens, utters_lens), columns=["persona_lens", "utters_lens"])
persona_df = pd.DataFrame(persona_lens, columns=["persona_lens"])
# %%
utters_df = pd.DataFrame(utters_lens, columns=["utters_lens"])
# %%

# %%
# lines = []
# [lines.extend(f.open().readlines()) for f in files]
# lines = [line.strip() for line in lines]
# lens = [len(line.split()) for line in lines]

# %%
persona_df.hist(column="persona_lens", bins=40)
# %%
utters_df.hist(column="utters_lens", bins=40)
