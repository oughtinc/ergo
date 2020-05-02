import os
from pathlib import Path

for file in Path(os.getcwd()).glob("*.ipynb"):
    os.system(f"jupytext --to md {Path(file).name}")
    os.system(f"jupytext --to notebook {Path(file).stem}.md")
    os.system(f"rm {Path(file).stem}.md")
