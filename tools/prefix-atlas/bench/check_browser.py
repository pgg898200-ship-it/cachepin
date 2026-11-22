"""Run budget and inverse-lookup interactions against the shipped calculator in Chrome."""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CHECK = """
<script>
(() => {
  const el = id => document.getElementById(id);
  const set = (id, value, event) => {
    el(id).value = value;
    el(id).dispatchEvent(new Event(event, {bubbles: true}));
  };
  const digits = id => el(id).textContent.replace(/[^0-9]/g,'');
  if (digits('hits') !== '183552') throw Error('Mooncake default');
  set('trace', 'harvard', 'change');
  if (digits('hits') !== '4471892') throw Error('FreeInference default');
  set('target', '80', 'input');
  el('apply').click();
  if (el('capacity').value !== '8822' || digits('hits') !== '4547539')
    throw Error('inverse capacity target');
  set('capacity', '0', 'input');
  if (digits('hits') !== '0') throw Error('zero cache');
  document.body.dataset.browserCheck = 'passed';
})();
</script>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chrome", default="google-chrome")
    args = p.parse_args()
    html = (ROOT / "demo/index.html").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ui.html"
        path.write_text(html.replace("</body>", CHECK + "</body>"), encoding="utf-8")
        cmd = [args.chrome, "--headless", "--no-sandbox", "--disable-gpu",
               "--user-data-dir=" + str(Path(tmp) / "profile"), "--dump-dom", path.as_uri()]
        result = subprocess.run(cmd, check=True, capture_output=True, timeout=40)
    if b'data-browser-check="passed"' not in result.stdout:
        raise SystemExit("Calculator interaction check failed")
    print("Calculator default workloads, inverse lookup and empty-cache interactions passed.")


if __name__ == "__main__":
    main()
