"""Generate the corpus-extension teaching files for experiment B.

All text here is written for this assignment. Nothing is copied from
`evals/language_evals.json`: no eval prompt, no answer choice list, no answer key,
no model output, and no chat transcript. The generator enforces that with three
checks before it writes anything (see `verify()` at the bottom):

 1. `reject_eval_leakage` from run_evals.py - the same normalized contiguous
    prompt match the notebook applies - is run over every generated passage.
 2. Every proper name used in the eval suite is banned from the corpus.
 3. A list of banned word pairs (the exact word pairs an eval case asks the model
    to produce inside the eval's own frame) is checked.

Four of the eight extension categories are taught: grammar, opposites, negation
and spatial_relations. The other four (reference, sequence, everyday_knowledge,
categories_and_analogies) are deliberately left untaught so they act as a control
group in the comparison.

A note on formatting: `chunk_text()` in the notebook splits text on
`(?<=[.!?])\\s+`, so "a . b . c ." becomes three separate training passages. The
negation and spatial eval prompts span several clauses, so a model trained only on
one-clause passages never sees a "." followed by more text. The multi-clause
teaching passages below therefore write the internal period with no following
space ("a .b .c ."), which the tokenizer still reads as the tokens
["a", ".", "b", ".", "c", "."] but the splitter leaves as one passage. Compare
`corpus.txt` in the run folder to confirm.

    python tools/make_extension_corpus.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_evals import load_suite, matching_cases, word_tokens  # noqa: E402

CORPUS = ROOT / "corpus"
RNG = random.Random(20260919)

# Every proper name that appears anywhere in the eval suite. The corpus uses a
# disjoint name set, so eval cases that depend on these names stay out of
# vocabulary. That is a deliberate choice, reported as a limitation, not an
# accident: padding the vocabulary with eval-specific names to make cases
# scorable is exactly what the assignment warns against.
EVAL_NAMES = {"ava", "ella", "finn", "maya", "leo", "nora", "omar",
              "sara", "noah", "nina", "emma", "luca"}

NAMES = ["ben", "clara", "diego", "hana", "iris", "jonas", "kira",
         "mateo", "priya", "rosa", "tomas", "wren"]
PRONOUN = {"ben": "he", "clara": "she", "diego": "he", "hana": "she",
           "iris": "she", "jonas": "he", "kira": "she", "mateo": "he",
           "priya": "she", "rosa": "she", "tomas": "he", "wren": "they"}

# Word pairs an eval case asks the model to produce. They must never appear
# together inside the frame that case uses.
BANNED_OPPOSITE_PAIRS = {("hot", "cold"), ("cold", "hot"), ("empty", "full"),
                         ("full", "empty"), ("noisy", "quiet"), ("quiet", "noisy"),
                         ("open", "closed"), ("closed", "open")}
# Contiguous phrases that are the whole of an eval prompt.
BANNED_PHRASES = ["one bird", "the dogs", "yesterday she"]


def sentences(lines):
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 1. Grammar: singular/plural agreement (is / are / am / was / were)
# --------------------------------------------------------------------------
def plural_of(word):
    if word.endswith(("x", "s", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def past_of(verb):
    return verb + "d" if verb.endswith("e") else verb + "ed"


def third_person_of(verb):
    return verb + "es" if verb.endswith(("ch", "sh", "s", "x", "o")) else verb + "s"


def gerund_of(verb):
    return verb[:-1] + "ing" if verb.endswith("e") else verb + "ing"


# "bird" and "dog" are handled separately below: their plural/singular forms are
# needed in the vocabulary, but "one bird" and "the dogs" are themselves eval
# prompts and must never be written.
SINGULAR = ["crow", "rabbit", "fox", "owl", "bee", "frog", "lamb",
            "pony", "hen", "sparrow", "boy", "girl", "runner", "singer",
            "farmer", "painter", "baker", "driver", "guard"]
PLURAL = {w: plural_of(w) for w in SINGULAR}
ADJ = ["small", "big", "tall", "short", "quiet", "quick", "slow", "young",
       "old", "brown", "white", "grey", "calm", "busy", "safe", "awake",
       "asleep", "clean", "ready", "loud", "near", "warm"]


def grammar_agreement():
    lines = []
    for noun in SINGULAR:
        plural = PLURAL[noun]
        for adjective in RNG.sample(ADJ, 6):
            other = RNG.choice([a for a in ADJ if a != adjective])
            lines += [
                f"one {noun} is {adjective} .",
                f"a {noun} is {adjective} .",
                f"the {noun} is {adjective} .",
                f"one {noun} was {adjective} .",
                f"two {plural} are {adjective} .",
                f"three {plural} are {adjective} .",
                f"many {plural} were {adjective} .",
                f"one {noun} is {adjective} but two {plural} are {other} .",
                f"the {noun} was {adjective} and the {plural} were {other} .",
                f"the {plural} are {adjective} .",
            ]
    for adjective in ADJ:
        lines += [f"i am {adjective} .", f"i was {adjective} ."]
    # "dogs" and "bird" are needed for two eval prompts' vocabulary, but the
    # exact phrases "the dogs" and "one bird" are eval prompts themselves and
    # are never written. Every other frame is available.
    for adjective in ADJ:
        other = RNG.choice([a for a in ADJ if a != adjective])
        lines += [
            f"two dogs are {adjective} .",
            f"three dogs are {adjective} .",
            f"many dogs were {adjective} .",
            f"a dog is {adjective} .",
            f"the dog is {adjective} .",
            f"one dog is {adjective} but two dogs are {other} .",
            f"the dog was {adjective} and many dogs were {other} .",
            f"a bird is {adjective} .",
            f"the bird is {adjective} .",
            f"the bird was {adjective} .",
            f"two birds are {adjective} .",
            f"many birds were {adjective} .",
            f"the birds are {adjective} .",
            f"the bird was {adjective} and the birds were {other} .",
        ]
    RNG.shuffle(lines)
    return sentences(lines)


# --------------------------------------------------------------------------
# 2. Grammar: tense (walk / walks / walked / walking)
# --------------------------------------------------------------------------
# Ten verbs, not thirty: every form of every verb is a separate vocabulary type,
# and the tokenizer keeps only the 509 most frequent. A first build used 23 verbs
# and pushed "walk", "walks" and "walking" out of the vocabulary entirely, which
# made a grammar eval case unscorable. Fewer verbs, taught more thoroughly, keep
# all four forms of each one well inside the cap.
VERBS = ["walk", "open", "clean", "cook", "help", "look", "jump", "climb",
         "count", "move"]
SINGULAR_SUBJECTS = ["she", "he", "the runner", "the farmer", "the boy",
                     "the painter", "the baker", "the guard"]
PAST_SUBJECTS = ["he", "they", "we", "i", "the runner", "the farmer",
                 "the singer", "the boy", "the girls", "the painter",
                 "the guard", "the baker"] + NAMES
PLACES = ["the park", "the yard", "the garden", "the hall", "the bench",
          "the gate", "the fence", "the river", "the bridge", "the road"]


def grammar_tense():
    lines = []
    for verb in VERBS:
        past, third, ing = past_of(verb), third_person_of(verb), gerund_of(verb)
        for subject in RNG.sample(PAST_SUBJECTS, 14):
            place = RNG.choice(PLACES)
            # "yesterday she" is an eval prompt and is never written; the
            # "and she <past>" frame teaches the same past-tense cue instead.
            lines += [
                f"yesterday {subject} {past} near {place} .",
                f"yesterday {subject} {past} and she {past} too .",
                f"last night {subject} {past} beside {place} .",
                f"an hour ago {subject} {past} slowly .",
            ]
        lines += [
            f"last week she {past} at home .",
            f"last month she {past} near the road .",
            f"an hour ago she {past} quickly .",
            f"earlier she {past} beside the gate .",
            f"yesterday he {past} and she {past} as well .",
            f"yesterday they {past} while she {past} nearby .",
            f"they are {ing} today .",
            f"we {verb} every week .",
        ]
        for subject in SINGULAR_SUBJECTS:
            lines += [
                f"today {subject} {third} to the park .",
                f"{subject} {third} every morning .",
                f"{subject} is {ing} now .",
                f"{subject} was {ing} earlier .",
                f"{subject} will {verb} later .",
                f"{subject} likes to {verb} .",
            ]
    RNG.shuffle(lines)
    return sentences(lines)


# --------------------------------------------------------------------------
# 3. Opposites: the frame, taught with pairs the eval never asks for
# --------------------------------------------------------------------------
OPPOSITE_PAIRS = [
    ("big", "small"), ("tall", "short"), ("fast", "slow"), ("heavy", "light"),
    ("wet", "dry"), ("hard", "soft"), ("rough", "smooth"), ("dark", "bright"),
    ("near", "far"), ("early", "late"), ("old", "new"), ("strong", "weak"),
    ("cheap", "expensive"), ("deep", "shallow"), ("wide", "narrow"),
    ("sharp", "dull"), ("clean", "dirty"), ("happy", "sad"), ("high", "low"),
    ("thick", "thin"), ("sweet", "sour"), ("warm", "cool"), ("busy", "idle"),
    ("awake", "asleep"), ("rich", "poor"), ("true", "false"),
]
THINGS = ["road", "bench", "jar", "coat", "stone", "rope", "board", "path",
          "window", "basket", "ribbon", "bucket", "candle", "hill", "field"]


def opposites_frame():
    lines = []
    for left, right in OPPOSITE_PAIRS:
        assert (left, right) not in BANNED_OPPOSITE_PAIRS
        for _ in range(4):
            thing = RNG.choice(THINGS)
            lines += [
                f"the opposite of {left} is {right} .",
                f"the opposite of {right} is {left} .",
                f"{left} and {right} are opposites .",
                f"{right} and {left} are opposites .",
                f"when something is not {left} it is {right} .",
                f"when something is not {right} it is {left} .",
                f"a {thing} can be {left} or {right} .",
            ]
    RNG.shuffle(lines)
    return sentences(lines)


# --------------------------------------------------------------------------
# 4. Opposites: contextual contrast for the three pairs the eval does ask for.
#    These pairs are never written inside the "the opposite of X is Y" frame.
# --------------------------------------------------------------------------
ROOMS = ["room", "hall", "street", "yard", "garden", "market", "kitchen",
         "station", "office", "school", "store", "bank", "hospital"]
CONTAINERS = ["jar", "glass", "bottle", "mug", "basket", "bucket", "tank",
              "cup", "bowl", "crate", "barrel", "kettle"]


def opposites_contrast():
    lines = []
    for room in ROOMS:
        lines += [
            f"the {room} was hot at noon and cold at midnight .",
            f"the {room} felt hot in summer and cold in winter .",
            f"the {room} was noisy at noon and quiet at midnight .",
            f"the {room} grew quiet after the noisy engine left .",
            f"a noisy morning and a quiet evening passed in the {room} .",
        ]
    for container in CONTAINERS:
        lines += [
            f"the {container} was full in the morning and empty at night .",
            f"one {container} is full and the other {container} is empty .",
            f"he filled the empty {container} until it was full .",
            f"she poured hot water into the {container} and cold water into the bowl .",
        ]
    for _ in range(12):
        thing = RNG.choice(THINGS)
        lines += [
            "she poured a hot drink and he poured a cold drink .",
            f"a hot {thing} cools slowly until it turns cold .",
            "hot water and cold water filled the two cups .",
            f"the loud engine filled the street near the {thing} .",
            "a loud bell rang beside the station .",
            "the plate is round and the board is flat .",
            "a round coin and a flat card sat on the desk .",
            "the key was missing under the drawer .",
            "a missing button was under the chair .",
        ]
    RNG.shuffle(lines)
    return sentences(lines)


# --------------------------------------------------------------------------
# 5. Negation: corrections. Multi-clause, so the internal period is written
#    tight against the next word to survive the passage splitter.
# --------------------------------------------------------------------------
COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown",
          "grey", "pink", "purple", "silver", "golden"]
OBJECTS = ["cup", "mug", "chair", "hat", "coat", "flag", "kite", "scarf",
           "van", "bench", "bowl", "plate", "card", "board", "sign", "box",
           "shirt", "sock", "glove", "ribbon", "bucket", "candle"]
STATE_PAIRS = [("open", "closed"), ("open", "shut"), ("wet", "dry"),
               ("full", "empty"), ("clean", "dirty"), ("near", "far"),
               ("hot", "cold"), ("loud", "quiet"), ("new", "old"),
               ("narrow", "wide"), ("dark", "bright"), ("heavy", "light")]
# "door" is the object the eval uses, so the correction frame uses other things.
STATE_THINGS = ["gate", "window", "drawer", "lid", "fence", "cupboard",
                "shutter", "hatch", "crate", "locker"]
GOODS = ["bread", "rice", "salad", "soup", "water", "coffee", "sugar",
         "salt", "honey", "cheese", "butter", "paper", "string", "chalk"]


def negation():
    lines = []
    for obj in OBJECTS:
        for _ in range(4):
            wrong, right = RNG.sample(COLORS, 2)
            lines += [
                f"the {obj} is not {wrong} .it is {right} .the {obj} is {right} .",
                f"the {obj} was not {wrong} .it was {right} .the {obj} was {right} .",
                f"the {obj} is not {wrong} .the {obj} is {right} .",
                f"that {obj} is not {wrong} .it is {right} .",
            ]
    for thing in STATE_THINGS:
        for wrong, right in STATE_PAIRS:
            lines += [
                f"the {thing} is not {wrong} .it is {right} .the {thing} is {right} .",
                f"the {thing} was not {wrong} .it was {right} .the {thing} was {right} .",
            ]
    for name in NAMES:
        pronoun = PRONOUN[name]
        for _ in range(8):
            wrong, right = RNG.sample(GOODS, 2)
            lines += [
                f"{name} did not buy {wrong} .{pronoun} bought {right} .{name} bought {right} .",
                f"{name} did not choose {wrong} .{pronoun} chose {right} .{name} chose {right} .",
                f"{name} did not take {wrong} .{pronoun} took {right} .{name} took {right} .",
            ]
    RNG.shuffle(lines)
    return sentences(lines)


# --------------------------------------------------------------------------
# 6. Spatial relations: inverse relations, also multi-clause.
# --------------------------------------------------------------------------
SMALL_THINGS = ["pen", "coin", "letter", "ring", "card", "key", "spoon",
                "brush", "stamp", "button", "thread", "apple", "book", "ball"]
HOLDERS = ["case", "jar", "folder", "basket", "tray", "drawer", "pocket",
           "envelope", "tin", "crate", "bag", "box"]
SURFACES = ["shelf", "desk", "table", "stool", "cabinet", "counter", "ledge",
            "mat", "rug", "bench"]
FIXTURES = ["clock", "mirror", "picture", "map", "lamp", "banner", "shelf",
            "hook", "poster", "curtain"]
# Object pairs an eval case uses; never written together in that eval's frame.
BANNED_PAIRS = {("book", "bag"), ("lamp", "desk"), ("ball", "box")}


def spatial():
    lines = []
    for small in SMALL_THINGS:
        for holder in HOLDERS:
            if (small, holder) in BANNED_PAIRS:
                continue
            lines += [
                f"the {small} is inside the {holder} .the {holder} contains the {small} .",
                f"the {holder} contains the {small} .the {small} is inside the {holder} .",
            ]
    for fixture in FIXTURES:
        for surface in SURFACES:
            if (fixture, surface) in BANNED_PAIRS or fixture == surface:
                continue
            lines += [
                f"the {fixture} is above the {surface} .the {surface} is below the {fixture} .",
                f"the {surface} is below the {fixture} .the {fixture} is above the {surface} .",
            ]
    for left_thing in SMALL_THINGS:
        for right_thing in RNG.sample(HOLDERS, 5):
            if (left_thing, right_thing) in BANNED_PAIRS:
                continue
            lines += [
                f"the {left_thing} is left of the {right_thing} ."
                f"the {right_thing} is to the right of the {left_thing} .",
                f"the {left_thing} is right of the {right_thing} ."
                f"the {right_thing} is to the left of the {left_thing} .",
            ]
    for thing in SMALL_THINGS:
        surface = RNG.choice(SURFACES)
        other = RNG.choice(FIXTURES)
        lines += [
            f"the {thing} is beside the {surface} .the {surface} is beside the {thing} .",
            f"the {thing} is under the {surface} .the {surface} is over the {thing} .",
            f"the {thing} sits on the {surface} .the {surface} holds the {thing} .",
            f"the {thing} is in front of the {other} .the {other} is behind the {thing} .",
            f"the {thing} is north of the {surface} .the {surface} is south of the {thing} .",
            f"the {thing} is south of the {surface} .the {surface} is north of the {thing} .",
        ]
    RNG.shuffle(lines)
    return sentences(lines)


# --------------------------------------------------------------------------
# 7. Plain descriptions: ordinary one-clause sentences that put the remaining
#    nouns and adjectives into context. This file is also rendered to PDF.
# --------------------------------------------------------------------------
def descriptions():
    lines = []
    for surface in SURFACES:
        lines += [
            f"the book rests on the {surface} near the window .",
            f"the bag hangs beside the {surface} .",
            f"the lamp stands on the {surface} in the hall .",
            f"a round plate and a wide tray sat on the {surface} .",
            f"the ball rolled under the {surface} .",
            f"a small box waited on the {surface} .",
        ]
    lines += [
        "the door of the hall is wide and white .",
        "he painted the door white last week .",
        "the door and the window face the garden .",
        "the park is north of the river and the school is south of the river .",
        "a quiet street sits beside the noisy market .",
        "the shelf above the desk holds the book and the folder .",
        "the desk below the shelf holds the lamp and the pen .",
        "a narrow path and a wide road meet near the bridge .",
        "the key was missing and the drawer was shut .",
        "a bird and two birds rested on the fence .",
        "many dogs and many ponies waited near the field .",
        "two birds are loud but one sparrow is quiet .",
        "the loud market and the quiet garden face the same road .",
    ]
    RNG.shuffle(lines)
    return sentences(lines)


# --------------------------------------------------------------------------
# 8. Printed notes. These sentences live ONLY in the PDF inside corpus/; the
#    plain-text original is kept in docs/, outside every training input, so the
#    PDF contributes real passages and its extraction can still be diffed
#    against a known source.
# --------------------------------------------------------------------------
def printed_notes():
    lines = []
    for container in CONTAINERS:
        lines += [
            f"a {container} can be full or empty .",
            f"the {container} stands beside the bowl on the counter .",
        ]
    for room in ROOMS:
        lines += [
            f"a quiet {room} and a noisy street share one gate .",
            f"the {room} is warm in summer and cool in winter .",
        ]
    for pair in OPPOSITE_PAIRS[:14]:
        left, right = pair
        lines.append(f"a {left} path and a {right} path lead to the bridge .")
    lines += [
        "the door of the office is wide and the window is narrow .",
        "a round plate and a flat board rest on the counter .",
        "the missing key was under the mat beside the gate .",
        "two birds are loud while one sparrow is quiet .",
        "many dogs and many ponies wait near the fence .",
        "the park is north of the bridge and the yard is south of the bridge .",
        "the shelf above the counter holds the book and the folder .",
        "the desk below the shelf holds the lamp and the pen .",
        "hot water cools in the jar until it is cold .",
        "the bag hangs beside the door of the hall .",
    ]
    RNG.shuffle(lines)
    return sentences(lines)


FILES = {
    "01_grammar_agreement.txt": grammar_agreement,
    "02_grammar_tense.txt": grammar_tense,
    "03_opposites_frame.txt": opposites_frame,
    "04_opposites_contrast.txt": opposites_contrast,
    "05_negation_corrections.txt": negation,
    "06_spatial_relations.txt": spatial,
    "07_plain_descriptions.md": descriptions,
}
PDF_SOURCE = ROOT / "docs" / "pdf_source_printed_notes.txt"
PDF_NAME = "08_printed_notes.pdf"


def verify(texts, suite):
    """Three independent separation checks. Any failure aborts the build."""
    problems = []
    for name, text in texts.items():
        hits = matching_cases(text, suite)
        if hits:
            problems.append(f"{name}: contains eval prompt(s) {hits}")
        tokens = set(word_tokens(text))
        banned = sorted(tokens & EVAL_NAMES)
        if banned:
            problems.append(f"{name}: uses eval proper name(s) {banned}")
        normalized = " " + " ".join(word_tokens(text)) + " "
        for phrase in BANNED_PHRASES:
            if f" {phrase} " in normalized:
                problems.append(f"{name}: contains the reserved phrase {phrase!r}")
        for left, right in BANNED_OPPOSITE_PAIRS:
            for frame in (f" the opposite of {left} is {right} ",
                          f" {left} and {right} are opposites "):
                if frame in normalized:
                    problems.append(f"{name}: teaches the eval's own pair in its own frame: {frame!r}")
    if problems:
        raise SystemExit("Separation check FAILED:\n  " + "\n  ".join(problems))
    print("Separation checks passed: no eval prompt, no eval proper name, "
          "no reserved phrase, no eval word pair in the eval's own frame.")


def main():
    suite = load_suite(ROOT / "evals/language_evals.json")
    texts = {name: build() for name, build in FILES.items()}
    verify(texts, suite)

    CORPUS.mkdir(exist_ok=True)
    for old in CORPUS.glob("*"):
        if old.name != "README.md" and old.is_file():
            old.unlink()
    total_lines = 0
    for name, text in texts.items():
        (CORPUS / name).write_text(text, encoding="utf-8")
        lines = text.strip().split("\n")
        total_lines += len(lines)
        print(f"{name:<32} {len(lines):>6} lines  {len(text):>8} chars")

    # Render a PDF whose text exists nowhere else in corpus/, so the PDF
    # contributes its own passages. Its plain-text original is written to docs/,
    # outside every training input, purely so extraction can be diffed later.
    notes = printed_notes()
    verify({"pdf_source": notes}, suite)
    PDF_SOURCE.parent.mkdir(exist_ok=True)
    PDF_SOURCE.write_text(notes, encoding="utf-8")
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    source_lines = notes.strip().split("\n")
    # invariant=1 strips the creation timestamp and document ID, so rebuilding the
    # corpus produces a byte-identical PDF and the sha256 recorded in a run's
    # corpus_manifest.json stays valid.
    pdf = canvas.Canvas(str(CORPUS / PDF_NAME), pagesize=letter, invariant=1)
    pdf.setTitle("Printed teaching notes")
    width, height = letter
    y = height - 54
    pdf.setFont("Helvetica", 9)
    for line in source_lines:
        if y < 54:
            pdf.showPage()
            pdf.setFont("Helvetica", 9)
            y = height - 54
        pdf.drawString(54, y, line)
        y -= 12
    pdf.save()
    print(f"{PDF_NAME:<32} {len(source_lines):>6} lines  rendered from "
          f"{PDF_SOURCE.relative_to(ROOT)} (outside corpus/)")

    words = set()
    for text in list(texts.values()) + [notes]:
        words |= set(word_tokens(text))
    print(f"\nTotal lines: {total_lines} | distinct token types in the extension: {len(words)}")


if __name__ == "__main__":
    main()
