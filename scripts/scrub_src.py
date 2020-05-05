import argparse
import json
from pathlib import Path
import subprocess

strip_metadata = {
    "jupytext": {"notebook_metadata_filter": "-all", "cell_metadata_filter": "-all"}
}

strip_metadata_string = json.dumps(strip_metadata)


def scrub(notebooks_path, scrubbed_path):
    for file_to_scrub in scrubbed_path.glob("*.ipynb"):
        scrubbed_file_stem = scrubbed_path / file_to_scrub.stem
        subprocess.run(
            f"jupytext --output '{scrubbed_file_stem}.md' --to md '{file_to_scrub}' --update-metadata '{strip_metadata_string}'",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        res = subprocess.run(
            f"jupytext --output '{file_to_scrub}' --to notebook '{scrubbed_file_stem}.md'",
            shell=True,
            check=True,
            universal_newlines=True,
            stdout=subprocess.PIPE,
        )
        print(res.stdout.split("\n")[1])
        subprocess.run(
            f"rm '{scrubbed_file_stem}.md'",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks_path", type=Path)
    parser.add_argument("scrubbed_path", type=Path)
    p = parser.parse_args()
    assert p.notebooks_path.is_dir()
    assert p.scrubbed_path.is_dir()
    scrub(p.notebooks_path, p.scrubbed_path)
