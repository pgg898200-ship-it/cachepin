"""Run the built CLI against tie, group, invalid-input, and CRLF cases."""
import csv
import io
from pathlib import Path
import subprocess

binary = Path(__file__).resolve().parents[1] / "racecurve"


def run(text, *args):
    return subprocess.run([binary, *(args or ("10", "5", "4"))], text=True,
                          input=text, capture_output=True)


r = run("group,primary_us,backup_us\r\n0,10,5\r\n0,12,5\r\n1,9,0\r\n")
assert r.returncode == 0, r.stderr
rows = [list(map(int, x.values())) for x in csv.DictReader(io.StringIO(r.stdout))]
assert rows == [
    [0, 3, 2, 3, 0, 0, 10, 10, 20],
    [5, 3, 2, 3, 0, 0, 25, 10, 35],
    [10, 3, 2, 1, 1, 1, 31, 2, 33],
    [15, 3, 2, 0, 1, 1, 31, 0, 31],
]
bad = ["", "primary_us,backup_us\n", "primary_us,backup_us\n-1,0",
       "primary_us,backup_us\n1,2,3", "primary_us,backup_us\n9,x",
       "group,primary_us,backup_us\n2,1,3\n1,3,5", "primary_us,backup_us\n2147483648,2"]
for text in bad:
    assert run(text).returncode != 0, text
for args in [("10", "0", "2"), ("10", "2", "0"), ("10", "2147483647", "3")]:
    assert run("primary_us,backup_us\n2,3\n", *args).returncode != 0
print("CLI checks passed.")
