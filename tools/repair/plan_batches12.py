"""Cleanup plan for batches 1-2 (ex 0001-0200), from the two check reports. Run:
python -m tools.repair.plan_batches12 --batch1 F1 --batch2 F2 --out OUT [--report R]"""
import argparse
import json
import os
import tempfile

from tools.repair.repair import repair

CONFIG10 = 'DEBUG = True\nPORT = 8080\nHOST = "localhost"\nDB_URL = "sqlite:///app.db"\nTIMEOUT = 30\nRETRIES = 3\n\n' \
           'def get_url():\n    return f"http://{HOST}:{PORT}"\n\n'
TRACE_NOTE = "x"

BATCH1 = {
    "remove": {"0002"},
    "instant": {"0068", "0069", "0075", "0078", "0081", "0084"},
    "patch": {
        "0027": [("assistant", "100°C", "100 degrees C")],
        "0032": [("assistant", "Tug'ilgan kuningiz bilan!", "Tug'ilgan kuningiz muborak!")],
        "0095": [("assistant", '</call>\nreturn "Hello, stranger"', '</call>\n        return "Hello, stranger"')],
        "0097": [("user", "in JavaScript", "in Python"), ("user", "`console.log(add(2,3))`", "`print(add(2,3))`"),
                 ("assistant", 'shell(node -e "console.log(add(2,3))")', 'shell(python -c "print(add(2,3))")'),
                 ("assistant", "function add(a, b) { return a + b; }", "def add(a, b): return a + b")],
    },
    "fixtures": {
        "0086": {}, "0089": {}, "0092": {}, "0096": {}, "0098": {}, "0099": {}, "0091": {}, "0097": {},
        "0087": {"main.py": "", "utils.py": "", "README.md": "", "assets/player.png": ""},
        "0088": {"main.py": "nums = [1, 2, 3]\n\nprint(nums)\nprint(total)\n"},
        "0090": {"config.py": CONFIG10},
        "0093": {"assets/player.png": "", "assets/enemy.png": "", "assets/jump.wav": "", "assets/level1.json": ""},
        "0094": {"settings.py": "".join("# section %d\n" % i for i in range(1, 20)) +
                 'MAX_USERS = 100\nRATE_LIMIT = 5\nLOG_LEVEL = "INFO"\n\ndef is_allowed(n):\n    return n < MAX_USERS\n'},
        "0095": {"app.py": "".join("# line %d\n" % i for i in range(1, 10)) +
                 'def greet(name):\n    if name is None:\n        return "Hello, " + name\n    return "Hello, " + name\n\n'
                 'print(greet(None))\n'},
        "0100": {"main.py": "", "app.py": "", "utils.py": "", "README.md": "", "tests/test_main.py": ""},
    },
}

BATCH2 = {
    "remove": {"0149", "0151", "0159", "0160", "0176", "0199"},
    "instant": {"0154", "0156", "0158", "0146"},
    "patch": {
        "0183": [("assistant", "\nReading handle_request will show where the exception is raised.", "")],
        "0197": [("assistant", "pygame is not installed. The tools cannot install packages. Note this",
                  "pygame is not installed. Note this")],
    },
    "rewrite": {
        "0166": (
            "Run main.py, fix any error, and make it print the sum of a list. If the same error appears twice, change your approach.",
            "<thinking> Run the file to see the first error. </thinking>\n"
            "<call> shell(python main.py) </call>\n<result> x </result>\n"
            "<thinking> A parenthesis is missing on line 2. Read the file. </thinking>\n"
            "<call> read(main.py, 1-4) </call>\n<result> x </result>\n"
            "<thinking> Line 2 is missing a closing parenthesis. Add it. </thinking>\n"
            "<call> edit(main.py, 2-2) </call>\ntotal = sum(nums)\n"
            "<call> shell(python main.py) </call>\n<result> x </result>\n"
            "<thinking> A different error now: totl on line 3 is a typo for total. It is not the same error, so fix the name. </thinking>\n"
            "<call> edit(main.py, 3-3) </call>\nprint(total)\n"
            "<call> shell(python main.py) </call>\n<result> x </result>\n"
            "<done> main.py prints 10\n"
            "The first run failed with an unclosed parenthesis on line 2. After closing it, a typo on line 3 (totl) raised a NameError. "
            "Fixing both makes main.py print 10."),
        "0200": (
            "Snake loyihasini yakunlang. Barcha fayllar yozilgan. Oxirgi marta ishga tushiring va natijani ayting.",
            "<call> shell(python main.py) </call>\n<result> x </result>\n"
            "<done> snake game runs\n<goal> snake game with score and collision\n"
            "<done> settings, snake movement, food, collision, score, assets\n<todo> nothing\n"
            "Dastur xatosiz ishga tushdi va hisob 0 ni chiqardi. Loyiha tayyor."),
    },
    "fixtures": {
        "0161": {}, "0163": {}, "0165": {}, "0167": {}, "0169": {}, "0172": {}, "0177": {}, "0180": {}, "0195": {},
        "0196": {},
        "0162": {"app.py": "import sys\nimport os\nfrom utils import helper\nprint(help_user(\"Alice\"))\n",
                 "utils.py": 'def help_user(name):\n    return "Hello, " + name\n'},
        "0164": {"main.py": 'from app import run\n\nif __name__ == "__main__":\n    run()\n',
                 "app.py": 'from utils import load_config\n\ndef run():\n    config = load_config()\n    print("Started with", config["mode"])\n',
                 "utils.py": 'def load_config():\n    return {"mode": "dev"}\n', "README.md": "# Demo\n"},
        "0166": {"main.py": "nums = [1, 2, 3, 4]\ntotal = sum(nums\nprint(totl)\n"},
        "0168": {"data.csv": "name,score\nali,80\nvali,90\nhasan,\nhusan,70\nolim,85\n"},
        "0170": {"main.py": "", "utils.py": "", "README.md": ""},
        "0173": {"config.py": CONFIG10},
        "0174": {"main.py": "nums = [1, 2, 3]\n\nprint(nums)\nprint(total)\n"},
        "0175": {"main.py": "", "app.py": "", "utils.py": "", "README.md": "", "tests/test_app.py": ""},
        "0178": {"app.py": '"""Prints the sum of 2 and 3."""\n\na = 2\nb = 3\nprint(a + b)\n'},
        "0179": {"test.py": "# test\n\nimport helper\nhelp.run()\n", "help.py": 'def main():\n    print("help")\n'},
        "0197": {"game.py": "import pygame\n\npygame.init()\n"},
        "0200": {"main.py": 'print("Score: 0")\n'},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch1", required=True)
    ap.add_argument("--batch2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="repair_")
    t1, r1 = repair(open(a.batch1).read(), BATCH1, work)
    t2, r2 = repair(open(a.batch2).read(), BATCH2, work)
    open(a.out, "w").write(t1 + "\n" + t2)
    if a.report:
        json.dump({"batch1": r1, "batch2": r2}, open(a.report, "w"), indent=1)
    print(json.dumps({k: {x: len(v) for x, v in r.items()} for k, r in (("b1", r1), ("b2", r2))}))


if __name__ == "__main__":
    main()
