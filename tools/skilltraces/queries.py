"""Questions to ask the real Wikipedia search. Only the QUESTIONS are written here; every answer comes from the API.
Each entry: (kind, english_name, uzbek_name_or_None, expected_wikipedia_title). The expected title says which returned
result answers the question, so the training sample is kept only when that result really came back."""
import random

COUNTRIES = [  # (country, uzbek country, capital = expected title)
    ("Uzbekistan", "O'zbekiston", "Tashkent"), ("France", "Fransiya", "Paris"), ("Japan", "Yaponiya", "Tokyo"), ("Turkey", "Turkiya", "Ankara"),
    ("Kazakhstan", "Qozog'iston", "Astana"), ("Australia", "Avstraliya", "Canberra"), ("Canada", "Kanada", "Ottawa"), ("Germany", "Germaniya", "Berlin"),
    ("Italy", "Italiya", "Rome"), ("Spain", "Ispaniya", "Madrid"), ("Russia", "Rossiya", "Moscow"), ("China", "Xitoy", "Beijing"),
    ("India", "Hindiston", "New Delhi"), ("Egypt", "Misr", "Cairo"), ("Brazil", "Braziliya", "Brasília"), ("Argentina", "Argentina", "Buenos Aires"),
    ("Kyrgyzstan", "Qirg'iziston", "Bishkek"), ("Tajikistan", "Tojikiston", "Dushanbe"), ("Turkmenistan", "Turkmaniston", "Ashgabat"),
    ("Afghanistan", "Afg'oniston", "Kabul"), ("Iran", "Eron", "Tehran"), ("Pakistan", "Pokiston", "Islamabad"), ("South Korea", "Janubiy Koreya", "Seoul"),
    ("Thailand", "Tailand", "Bangkok"), ("Vietnam", "Vetnam", "Hanoi"), ("Indonesia", "Indoneziya", "Jakarta"), ("Poland", "Polsha", "Warsaw"),
    ("Ukraine", "Ukraina", "Kyiv"), ("Greece", "Gresiya", "Athens"), ("Portugal", "Portugaliya", "Lisbon"), ("Netherlands", "Niderlandiya", "Amsterdam"),
    ("Sweden", "Shvetsiya", "Stockholm"), ("Norway", "Norvegiya", "Oslo"), ("Finland", "Finlyandiya", "Helsinki"), ("Austria", "Avstriya", "Vienna"),
    ("Switzerland", "Shveytsariya", "Bern"), ("Mexico", "Meksika", "Mexico City"), ("Peru", "Peru", "Lima"), ("Chile", "Chili", "Santiago"),
    ("Kenya", "Keniya", "Nairobi"), ("Morocco", "Marokash", "Rabat"), ("Saudi Arabia", "Saudiya Arabistoni", "Riyadh"), ("Azerbaijan", "Ozarbayjon", "Baku"),
    ("Georgia", None, "Tbilisi"), ("Mongolia", "Mongoliya", "Ulaanbaatar"), ("Nepal", "Nepal", "Kathmandu"), ("Ireland", "Irlandiya", "Dublin"),
]
PEOPLE = [  # (name, expected title)
    "Albert Einstein", "Isaac Newton", "Marie Curie", "Alan Turing", "Ada Lovelace", "Charles Darwin", "Nikola Tesla", "Galileo Galilei",
    "Leonardo da Vinci", "Aristotle", "Plato", "Socrates", "Archimedes", "Euclid", "Pythagoras", "Johannes Kepler", "Nicolaus Copernicus",
    "Stephen Hawking", "Richard Feynman", "Niels Bohr", "Max Planck", "Michael Faraday", "James Clerk Maxwell", "Louis Pasteur", "Gregor Mendel",
    "Guido van Rossum", "Linus Torvalds", "Dennis Ritchie", "Ken Thompson", "Tim Berners-Lee", "Grace Hopper", "Claude Shannon", "John von Neumann",
    "Donald Knuth", "Edsger W. Dijkstra", "Bjarne Stroustrup", "James Gosling", "Brendan Eich", "Charles Babbage", "George Boole",
    "Timur", "Ulugh Beg", "Avicenna", "Al-Biruni", "Al-Farabi", "Ali-Shir Nava'i", "Babur", "Muhammad ibn Musa al-Khwarizmi", "Omar Khayyam",
    "Rumi", "William Shakespeare", "Leo Tolstoy", "Mark Twain", "Jane Austen", "Napoleon", "Julius Caesar", "Genghis Khan", "Alexander the Great",
    "Abraham Lincoln", "Mahatma Gandhi", "Nelson Mandela", "Winston Churchill",
]
CONCEPTS = [  # (english title, uzbek word or None)
    ("Photosynthesis", "fotosintez"), ("Gravity", "tortishish kuchi"), ("DNA", "DNK"), ("Black hole", "qora tuynuk"), ("Volcano", "vulqon"),
    ("Earthquake", "zilzila"), ("Electricity", "elektr"), ("Atom", "atom"), ("Molecule", "molekula"), ("Evolution", "evolyutsiya"),
    ("Inflation", "inflyatsiya"), ("Democracy", "demokratiya"), ("Algorithm", "algoritm"), ("Compiler", "kompilyator"), ("Operating system", "operatsion tizim"),
    ("Machine learning", "mashinali o'rganish"), ("Neural network", "neyron tarmoq"), ("Internet", "internet"), ("Transistor", "tranzistor"), ("Prime number", "tub son"),
    ("Pythagorean theorem", "Pifagor teoremasi"), ("Calculus", "hisob"), ("Probability", "ehtimollik"), ("Binary number", "ikkilik sanoq tizimi"), ("Entropy", None),
    ("Quantum mechanics", None), ("General relativity", None), ("Big Bang", None), ("Solar System", None), ("Milky Way", None),
    ("Plate tectonics", None), ("Greenhouse effect", None), ("Water cycle", None), ("Mitochondrion", None), ("Cell (biology)", None),
    ("Vaccine", None), ("Antibiotic", None), ("Silk Road", None), ("Renaissance", None), ("Industrial Revolution", None),
    ("Cold War", None), ("Great Wall of China", None), ("Pyramids of Giza", None), ("Mount Everest", None), ("Amazon River", None),
    ("Sahara", None), ("Aral Sea", None), ("Pacific Ocean", None), ("Antarctica", None), ("Python (programming language)", None),
    ("Linux", None), ("Git", None), ("Unix", None), ("HTTP", None), ("TCP/IP", None), ("Hash function", None), ("Binary search algorithm", None),
    ("Sorting algorithm", None), ("Recursion", None), ("Turing machine", None), ("Moore's law", None),
]
INVENTED_SYL = ["zor", "vel", "quin", "brax", "thal", "myr", "kos", "dun", "pheo", "lix", "nar", "tek", "vor", "ulm", "zyn", "grel", "ixo", "bam"]

EN_TEMPLATES = {
    "capital": ["What is the capital of {x}?", "capital of {x}?", "Which city is the capital of {x}?", "what's {x}'s capital", "Tell me the capital of {x}."],
    "person": ["Who is {x}?", "Who was {x}?", "tell me about {x}", "{x} - who is that?", "What is {x} known for?"],
    "concept": ["What is {x}?", "explain {x}", "what does {x} mean", "Tell me about {x}.", "{x}?"],
}
UZ_TEMPLATES = {
    "capital": ["{x} poytaxti qaysi shahar?", "{x}ning poytaxti nima?", "{x} poytaxti qayer?"],
    "person": ["{x} kim?", "{x} haqida ayt.", "{x} nima bilan mashhur?"],
    "concept": ["{x} nima?", "{x} haqida tushuntir.", "{x} degani nima?"],
}


def invented(rng, n):
    out = set()
    while len(out) < n:
        out.add("".join(rng.choice(INVENTED_SYL) for _ in range(rng.choice([2, 3]))) + rng.choice(["", "-%d" % rng.randint(2, 99)]))
    return sorted(out)


def all_queries(seed=0):
    """-> list of dicts {kind, query, expected, en: [questions], uz: [questions]} (invented ones have expected None)."""
    rng = random.Random(seed)
    items = []
    for en, uz, cap in COUNTRIES:
        items.append({"kind": "capital", "query": "capital of %s" % en, "expected": cap,
                      "en": [t.format(x=en) for t in EN_TEMPLATES["capital"]], "uz": [t.format(x=uz) for t in UZ_TEMPLATES["capital"]] if uz else []})
    for p in PEOPLE:
        items.append({"kind": "person", "query": p, "expected": p, "en": [t.format(x=p) for t in EN_TEMPLATES["person"]],
                      "uz": [t.format(x=p) for t in UZ_TEMPLATES["person"]]})
    for en, uz in CONCEPTS:
        name = en.split(" (")[0]
        items.append({"kind": "concept", "query": name, "expected": en, "en": [t.format(x=name) for t in EN_TEMPLATES["concept"]],
                      "uz": [t.format(x=uz) for t in UZ_TEMPLATES["concept"]] if uz else []})
    for w in invented(rng, 60):
        items.append({"kind": "invented", "query": w, "expected": None,
                      "en": [t.format(x=w) for t in ["What is {x}?", "Who invented {x}?", "Tell me about {x}.", "how do I use {x}?", "{x} meaning?"]],
                      "uz": [t.format(x=w) for t in ["{x} nima?", "{x} haqida ayt.", "{x} qanday ishlatiladi?"]]})
    return items
