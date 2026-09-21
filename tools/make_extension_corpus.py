"""Generate corpus-extension teaching files, with enforced eval separation.

All text is written for this assignment. Nothing is copied from
`evals/language_evals.json`: no eval prompt, no answer choice list, no answer key,
no model output, no chat transcript.

Five checks run before anything is written. The build aborts on any failure.

 1. `reject_eval_leakage` from run_evals.py - the notebook's own normalized
    contiguous prompt match - finds no eval prompt in any generated file.
 2. No proper name used anywhere in the eval suite appears.
 3. None of the reserved phrases (each of which is an entire eval prompt) appears.
 4. No word pair an eval asks for is written inside that eval's own frame.
 5. ANSWER-CONTINUATION GUARD. For every one of the 48 cases, find the longest run
    of tokens ending the prompt that also appears in the generated passages, then
    look at what follows it. If a suffix of >= MIN_GUARDED_SUFFIX tokens is followed
    by that case's answer more than MAX_ANSWER_SHARE of the time, the build fails.

Check 5 is the one that matters, and it is stricter than the upstream checker.
The upstream check only catches a *whole* prompt appearing verbatim. It passed
happily on an earlier version of this file that contained

    the clock is above the desk . the desk is below the clock .

which is not the eval prompt (the eval uses "lamp"), but shares an 8-token suffix
with it and is always followed by the answer. A model can score that case by
recalling a continuation instead of applying the relation. Frames whose answer is
a *relation word* (below, right) are especially exposed, because the answer does
not change when the nouns change - so for those frames the nouns the eval uses are
excluded entirely, and the model has to transfer the relation to them.

    python tools/make_extension_corpus.py --categories grammar,opposites,negation,spatial_relations
    python tools/make_extension_corpus.py --categories all --out corpus_seven
"""
import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_evals import load_suite, matching_cases, word_tokens  # noqa: E402

MIN_GUARDED_SUFFIX = 3
MAX_ANSWER_SHARE = 0.5
# Paraphrase guard: no single passage may carry more than this share of a case's
# content words AND its answer. Cases with fewer than MIN_CONTENT_WORDS content
# words are exempt - any legitimate sentence using "dogs" covers 100% of the
# content of the prompt "the dogs" - and are protected by BANNED_PHRASES instead.
MAX_CONTENT_COVERAGE = 0.75
MIN_CONTENT_WORDS = 3
# An answer key is the choice list; a passage reciting this many of a case's four
# choices is reciting it. Ordered-subsequence coverage catches a prompt rebuilt
# with words inserted between its tokens.
MIN_CHOICES_RECITED = 3
MAX_SUBSEQUENCE_COVERAGE = 0.8
# The PROVIDED classroom corpus shares runs of up to 7 tokens with its own eval
# prompts ("the team discussed the {noun} and the {context} at the {place} ."
# against the domain_place cases). That is the assignment's own baseline, and it
# is the standard this teaching material holds itself to: no passage written here
# may share a longer contiguous run with any eval prompt than the starter corpus
# already does. Teaching a frame necessarily shares the frame; it must not also
# share the frame's specific fillers.
MAX_SHARED_RUN = 7
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "am", "be", "been", "of", "to",
    "in", "on", "at", "and", "or", "but", "it", "its", "this", "that", "these",
    "those", "for", "with", "as", "by", "from", "we", "he", "she", "they", "i",
    "you", "his", "her", "their", "our", "not", "no", "so", "then", "than", "up",
    "down", "out", ".", ",", "?", "!", ";", ":", "'", "-",
}

# Every proper name that appears anywhere in the eval suite. The corpus uses a
# disjoint set, so cases that hinge on these names stay out of vocabulary. That is
# deliberate and reported as a limitation: padding the vocabulary with the exam's
# own names to make cases scorable is what the assignment warns against.
EVAL_NAMES = {"ava", "ella", "finn", "maya", "leo", "nora", "omar",
              "sara", "noah", "nina", "emma", "luca"}

# Nouns the eval uses inside frames whose answer is a relation word. Excluded from
# every relational template so no long eval-shaped context can be memorised.
EVAL_FRAME_NOUNS = {"book", "bag", "lamp", "shelf", "desk", "ball", "box", "door"}

BANNED_OPPOSITE_PAIRS = {("hot", "cold"), ("cold", "hot"), ("empty", "full"),
                         ("full", "empty"), ("noisy", "quiet"), ("quiet", "noisy"),
                         ("open", "closed"), ("closed", "open")}
BANNED_PHRASES = ["one bird", "the dogs", "yesterday she"]

NAMES = ["ben", "clara", "diego", "hana", "iris", "jonas", "kira", "mateo"]
PRONOUN = {"ben": "he", "clara": "she", "diego": "he", "hana": "she",
           "iris": "she", "jonas": "he", "kira": "she", "mateo": "he"}


def sentences(lines):
    return "\n".join(lines) + "\n"


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


# ==========================================================================
# grammar
# ==========================================================================
SINGULAR = ["crow", "rabbit", "fox", "owl", "frog", "lamb", "pony", "hen",
            "cat", "duck", "goat", "horse", "boy", "girl", "runner", "farmer"]
PLURAL = {w: plural_of(w) for w in SINGULAR}
ADJ = ["small", "big", "tall", "short", "quiet", "slow", "old", "brown",
       "white", "awake", "asleep", "clean", "loud", "warm"]
VERBS = ["walk", "open", "clean", "cook", "help", "look", "jump", "climb",
         "count", "move"]
SINGULAR_SUBJECTS = ["she", "he", "the runner", "the farmer", "the boy", "the girl"]
PAST_SUBJECTS = ["he", "they", "we", "i", "the runner", "the farmer",
                 "the boy", "the girls"] + NAMES
PLACES = ["the park", "the yard", "the garden", "the hall", "the gate",
          "the fence", "the river", "the bridge", "the road"]


def grammar_agreement(rng):
    lines = []
    for noun in SINGULAR:
        plural = PLURAL[noun]
        for adjective in rng.sample(ADJ, 6):
            other = rng.choice([a for a in ADJ if a != adjective])
            lines += [
                f"one {noun} is {adjective} .",
                f"a {noun} is {adjective} .",
                f"the {noun} is {adjective} .",
                f"one {noun} was {adjective} .",
                f"two {plural} are {adjective} .",
                f"three {plural} are {adjective} .",
                f"many {plural} were {adjective} .",
                f"the {plural} are {adjective} .",
                f"one {noun} is {adjective} but two {plural} are {other} .",
                f"the {noun} was {adjective} and the {plural} were {other} .",
            ]
    for adjective in ADJ:
        lines += [f"i am {adjective} .", f"i was {adjective} ."]
    # "dogs" and "bird" are needed for two eval prompts' vocabulary, but the exact
    # phrases "the dogs" and "one bird" are eval prompts and are never written.
    for adjective in ADJ:
        other = rng.choice([a for a in ADJ if a != adjective])
        lines += [
            f"two dogs are {adjective} .", f"three dogs are {adjective} .",
            f"many dogs were {adjective} .", f"a dog is {adjective} .",
            f"the dog is {adjective} .",
            f"one dog is {adjective} but two dogs are {other} .",
            f"the dog was {adjective} and many dogs were {other} .",
            f"a bird is {adjective} .", f"the bird is {adjective} .",
            f"the bird was {adjective} .", f"two birds are {adjective} .",
            f"many birds were {adjective} .", f"the birds are {adjective} .",
            f"the bird was {adjective} and the birds were {other} .",
        ]
    rng.shuffle(lines)
    return sentences(lines)


def grammar_tense(rng):
    lines = []
    for verb in VERBS:
        past, third, ing = past_of(verb), third_person_of(verb), gerund_of(verb)
        for subject in rng.sample(PAST_SUBJECTS, 12):
            place = rng.choice(PLACES)
            # "yesterday she" is an eval prompt. Banning the contiguous phrase is
            # not enough: "yesterday clara walked and she walked too ." contains
            # the same two tokens in order with a gap, and is followed by the
            # answer. No frame may pair "yesterday" with "she" at any distance, so
            # "she" is taught under other past-time cues and the model has to
            # transfer from "yesterday he/they/we" to "yesterday she".
            lines += [
                f"yesterday {subject} {past} near {place} .",
                f"last night {subject} {past} and she {past} too .",
                f"last night {subject} {past} beside {place} .",
                f"an hour ago {subject} {past} slowly .",
            ]
        lines += [
            f"last week she {past} near the gate .", f"last month she {past} near the road .",
            f"an hour ago she {past} quickly .", f"earlier she {past} beside the gate .",
            f"last week he {past} and she {past} as well .",
            f"last month they {past} while she {past} nearby .",
            f"they are {ing} today .", f"we {verb} every week .",
        ]
        for subject in SINGULAR_SUBJECTS:
            lines += [
                f"today {subject} {third} to the park .",
                f"{subject} {third} every morning .",
                f"{subject} is {ing} now .", f"{subject} was {ing} earlier .",
                f"{subject} will {verb} later .", f"{subject} likes to {verb} .",
            ]
    rng.shuffle(lines)
    return sentences(lines)


# ==========================================================================
# opposites
# ==========================================================================
OPPOSITE_PAIRS = [
    ("big", "small"), ("tall", "short"), ("fast", "slow"), ("heavy", "light"),
    ("wet", "dry"), ("hard", "soft"), ("dark", "bright"), ("near", "far"),
    ("early", "late"), ("old", "new"), ("wide", "narrow"),
    ("clean", "dirty"), ("warm", "cool"), ("awake", "asleep"),
]
THINGS = ["road", "bench", "jar", "coat", "stone", "rope", "path", "hill"]
ROOMS = ["room", "hall", "street", "yard", "garden", "market", "kitchen",
         "station", "office", "school", "store", "bank", "hospital"]
CONTAINERS = ["jar", "glass", "bottle", "mug", "basket", "bucket", "cup", "bowl"]


def opposites_frame(rng):
    lines = []
    for left, right in OPPOSITE_PAIRS:
        assert (left, right) not in BANNED_OPPOSITE_PAIRS
        for _ in range(4):
            thing = rng.choice(THINGS)
            lines += [
                f"the opposite of {left} is {right} .",
                f"the opposite of {right} is {left} .",
                f"{left} and {right} are opposites .",
                f"{right} and {left} are opposites .",
                f"when something is not {left} it is {right} .",
                f"when something is not {right} it is {left} .",
                f"a {thing} can be {left} or {right} .",
            ]
    rng.shuffle(lines)
    return sentences(lines)


def opposites_contrast(rng):
    """hot/cold, empty/full and noisy/quiet: the three pairs the eval asks about.
    They are taught ONLY as contextual contrasts, never inside the eval's frame.
    "round" and "flat" are here too - "round" is a distractor in one case, and it
    has to be frequent enough to survive the 509-type vocabulary cap."""
    lines = []
    for room in ROOMS:
        lines += [
            f"the {room} was hot at noon and cold at midnight .",
            f"the {room} felt hot in summer and cold in winter .",
            f"the {room} was noisy at noon and quiet at midnight .",
            f"the {room} grew quiet after the noisy engine left .",
            f"a noisy morning and a quiet evening are in the {room} .",
            f"the noisy {room} and the quiet garden face one road .",
            f"a round plate and a flat board are in the {room} .",
            f"the round coin and the flat card are in the {room} .",
        ]
    for container in CONTAINERS:
        lines += [
            f"the {container} was full in the morning and empty at night .",
            f"one {container} is full and one {container} is empty .",
            f"he filled the empty {container} until it was full .",
            f"hot water is in the {container} and cold water is in the bowl .",
            f"the {container} is round and the board is flat .",
            f"a round {container} and a flat plate are clean .",
        ]
    for thing in THINGS:
        lines += [
            f"a hot {thing} cools slowly and is cold .",
            f"the loud engine filled the street near the {thing} .",
            f"a round {thing} and a flat {thing} are different .",
            f"the missing key was under the {thing} .",
            f"a missing button was under the {thing} .",
        ]
    lines += [
        "hot water and cold water are on the table .",
        "hot water and cold water are in the bowl and the jar .",
        "a loud engine is beside the station .",
    ] * 4
    rng.shuffle(lines)
    return sentences(lines)
# ==========================================================================
# negation
# ==========================================================================
# Ordered so that `blue` follows `red`: the cyclic successor pairs removed in
# negation() then include the eval's own (red -> blue) pair.
COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown", "grey"]
# "box" and "door" are the nouns the eval uses; they are kept out of the frames.
OBJECTS = ["cup", "mug", "chair", "hat", "coat", "flag", "kite", "scarf",
           "van", "bench", "bowl", "plate", "card", "sign"]
# ("open", "closed") is the eval's own pair and is never written; open/shut and
# shut/closed carry the same idea without reproducing the eval's frame fillers.
STATE_PAIRS = [("shut", "closed"), ("closed", "shut"), ("open", "shut"), ("wet", "dry"),
               ("full", "empty"), ("clean", "dirty"), ("near", "far"),
               ("hot", "cold"), ("loud", "quiet"), ("new", "old"),
               ("narrow", "wide"), ("dark", "bright"), ("heavy", "light")]
STATE_THINGS = ["gate", "window", "drawer", "lid", "fence", "cupboard"]
GOODS = ["bread", "rice", "soup", "water", "coffee", "salt"]


def negation(rng):
    """Every object is paired with every ordered colour pair EXCEPT one cyclic
    successor pair per colour. Using all pairs removes any object-to-colour
    association and makes every colour equally frequent in the 'corrected to'
    slot, so the only way to answer is to copy the correction. Removing one
    successor pair per colour keeps that balance exact (every colour appears in
    every slot the same number of times) while dropping the eval's own pair
    red -> blue, which would otherwise give a 9-token run shared with an eval
    prompt - longer than the provided classroom corpus manages."""
    lines = []
    skip = {(COLORS[i], COLORS[(i + 1) % len(COLORS)]) for i in range(len(COLORS))}
    assert ("red", "blue") in skip, "the colour order must put blue after red"
    for obj in OBJECTS:
        for wrong in COLORS:
            for right in COLORS:
                if wrong == right or (wrong, right) in skip:
                    continue
                lines.append(f"the {obj} is not {wrong} .it is {right} .the {obj} is {right} .")
    for obj in rng.sample(OBJECTS, 8):
        for wrong, right in [rng.sample(COLORS, 2) for _ in range(6)]:
            lines += [
                f"the {obj} was not {wrong} .it was {right} .the {obj} was {right} .",
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
            wrong, right = rng.sample(GOODS, 2)
            lines += [
                f"{name} did not buy {wrong} .{pronoun} bought {right} .{name} bought {right} .",
                f"{name} did not choose {wrong} .{pronoun} chose {right} .{name} chose {right} .",
            ]
    rng.shuffle(lines)
    return sentences(lines)


# ==========================================================================
# spatial relations
# ==========================================================================
SMALL_THINGS = ["pen", "coin", "letter", "ring", "card", "key", "spoon", "brush", "stamp"]
HOLDERS = ["case", "jar", "folder", "basket", "tray", "drawer", "pocket", "tin"]
SURFACES = ["table", "stool", "cabinet", "counter", "ledge", "mat", "bench"]
FIXTURES = ["clock", "mirror", "picture", "map", "banner", "hook", "poster"]


def _safe(*words):
    return not (set(words) & EVAL_FRAME_NOUNS)


def spatial_unpaired(rng):
    """Experiment E's variant. Every relational passage in spatial() states a
    relation AND its inverse, so `left` and `right` (and above/below) are almost
    perfectly co-distributed: nearly every passage containing one contains the
    other. Next-token prediction then has little pressure to separate them, and
    in experiment D the nearest neighbour of `right` is `left` at cosine 0.76 -
    the mechanical reason the left/right eval case is a near-tie.

    This variant adds single-relation passages that mention one direction word
    without its inverse, so the two words stop appearing in lockstep. The paired
    passages are kept, because they are what teaches the inverse in the first
    place; only the co-distribution is broken."""
    lines = spatial(rng).strip().split("\n")
    singles = []
    for small in SMALL_THINGS:
        for holder in HOLDERS:
            if not _safe(small, holder):
                continue
            singles += [
                f"the {small} is left of the {holder} .",
                f"the {holder} is right of the {small} .",
                f"the {small} is inside the {holder} .",
            ]
    for fixture in FIXTURES:
        for surface in SURFACES:
            if not _safe(fixture, surface) or fixture == surface:
                continue
            singles += [
                f"the {fixture} is above the {surface} .",
                f"the {surface} is below the {fixture} .",
            ]
    for thing in SMALL_THINGS:
        for surface in SURFACES:
            if not _safe(thing, surface):
                continue
            singles += [
                f"the {thing} is north of the {surface} .",
                f"the {surface} is south of the {thing} .",
                f"the {thing} is under the {surface} .",
            ]
    lines += singles
    rng.shuffle(lines)
    return sentences(lines)


def spatial(rng):
    """The eval's own nouns are excluded from every relational frame. For the inverse
    relations the answer is a relation word that does not change with the nouns, so
    keeping the eval's nouns out is the only way to stop the case being answerable
    from a memorised continuation.

    Every relation is emitted symmetrically: each direction word appears exactly as
    often as its inverse, and every object pair appears in both orders. An earlier
    version sampled the pairs, which left `left` and `right` at different frequencies
    and let the model lean on a prior instead of resolving the direction."""
    lines = []
    for small in SMALL_THINGS:
        for holder in HOLDERS:
            if not _safe(small, holder):
                continue
            lines += [
                f"the {small} is inside the {holder} .the {holder} contains the {small} .",
                f"the {holder} contains the {small} .the {small} is inside the {holder} .",
            ]
    for fixture in FIXTURES:
        for surface in SURFACES:
            if not _safe(fixture, surface) or fixture == surface:
                continue
            lines += [
                f"the {fixture} is above the {surface} .the {surface} is below the {fixture} .",
                f"the {surface} is below the {fixture} .the {fixture} is above the {surface} .",
                f"the {fixture} hangs above the {surface} .the {surface} sits below the {fixture} .",
                f"the {surface} sits below the {fixture} .the {fixture} hangs above the {surface} .",
            ]
    for left_thing in SMALL_THINGS:
        for right_thing in HOLDERS:
            if not _safe(left_thing, right_thing):
                continue
            lines += [
                f"the {left_thing} is left of the {right_thing} ."
                f"the {right_thing} is to the right of the {left_thing} .",
                f"the {left_thing} is right of the {right_thing} ."
                f"the {right_thing} is to the left of the {left_thing} .",
                f"the {right_thing} is left of the {left_thing} ."
                f"the {left_thing} is to the right of the {right_thing} .",
                f"the {right_thing} is right of the {left_thing} ."
                f"the {left_thing} is to the left of the {right_thing} .",
            ]
    for thing in SMALL_THINGS:
        for surface in SURFACES:
            if not _safe(thing, surface):
                continue
            lines += [
                f"the {thing} is beside the {surface} .the {surface} is beside the {thing} .",
                f"the {thing} is under the {surface} .the {surface} is over the {thing} .",
                f"the {surface} is over the {thing} .the {thing} is under the {surface} .",
                f"the {thing} sits on the {surface} .the {surface} holds the {thing} .",
                f"the {thing} is north of the {surface} .the {surface} is south of the {thing} .",
                f"the {surface} is north of the {thing} .the {thing} is south of the {surface} .",
            ]
        for other in FIXTURES:
            if not _safe(thing, other):
                continue
            lines += [
                f"the {thing} is in front of the {other} .the {other} is behind the {thing} .",
                f"the {other} is in front of the {thing} .the {thing} is behind the {other} .",
            ]
    rng.shuffle(lines)
    return sentences(lines)
# ==========================================================================
# sequence  (optional third experiment)
# ==========================================================================
SEQ_VERBS = ["fold", "lift", "stack", "sort", "pack", "seal", "fill", "buy"]
# "breakfast" (lang_38) and "train"/"bus" (lang_39) are the eval's own nouns for
# these frames, so they are excluded here exactly as EVAL_FRAME_NOUNS is for the
# spatial frames. They still reach the vocabulary through ordinary sentences.
MEALS = ["lunch", "dinner", "supper", "brunch"]
VEHICLES = ["car", "taxi", "truck", "van"]


def sequence(rng):
    """"dry" and "wash" never appear in the slot the eval reads, "breakfast" never
    follows "after", and the eval's own vehicle pair is excluded, so no case is
    answerable by recalling a continuation."""
    lines = []
    trays = ["tray", "crate", "folder", "basket", "card"]
    for first in SEQ_VERBS:
        for second in SEQ_VERBS:
            if first == second:
                continue
            item = rng.choice(trays)
            lines += [
                f"first {first} the {item} .then {second} it .the last action is {second} .",
                f"first {first} the {item} .then {second} it .the first action is {first} .",
            ]
    for early in MEALS:
        for late in MEALS:
            if early == late:
                continue
            lines += [
                f"{late} happens after {early} .the earlier meal is {early} .",
                f"{early} happens before {late} .the later meal is {late} .",
            ]
    for meal in MEALS:
        for place in ROOMS[:5]:
            lines += [f"breakfast is a meal and so is {meal} .",
                      f"breakfast and {meal} are in the {place} .",
                      f"we wash the cup before breakfast in the {place} ."]
    for item in trays:
        lines += [f"we wash the {item} and then we dry it .",
                  f"she will wash and dry the {item} .",
                  f"the dry {item} and the wet {item} are clean .",
                  f"he will fill the {item} and then seal it ."]
    for early in VEHICLES:
        for late in VEHICLES:
            if early == late:
                continue
            # Three phrasings of the same idea. A single fixed phrasing shared an
            # 8-token run with the eval prompt; rotating them keeps the longest
            # shared run at or under the classroom corpus's own level.
            lines += [
                f"the {early} arrived before the {late} .the later vehicle was the {late} .",
                f"the {late} arrived after the {early} .the earlier vehicle was the {early} .",
                f"the {early} came before the {late} .the vehicle arriving later was the {late} .",
                f"the {late} came after the {early} .the vehicle arriving earlier was the {early} .",
            ]
    lines += [f"the train and the bus arrived at the station ." ,
              f"a bus and a train are each a vehicle ."] * 12
    rng.shuffle(lines)
    return sentences(lines)
# ==========================================================================
# everyday knowledge  (optional third experiment)
# ==========================================================================
def everyday(rng):
    """Templated so each fact appears in many distinct passages. The facts may
    overlap with the eval - the assignment permits that - but the eval's own
    phrasings never appear followed by the answer: "freezes into", "to stay" and
    "turn on a" are all avoided, and "stay" is taught with several continuations
    so it predicts nothing on its own."""
    lines = []
    for container in CONTAINERS:
        for place in ROOMS[:6]:
            lines += [
                f"cold water in the {container} freezes and the ice is hard .",
                f"the water freezes at night and the ice fills the {container} .",
                f"ice is cold water and the {container} of ice is hard .",
                f"hot water in the {container} is steam in the {place} .",
                f"the {place} was cold so the water was ice .",
                # "sand and wood and ice" would recite three of one eval case's
                # four answer choices in a single passage - an answer list.
                f"sand is on the path near the {place} .",
                f"the wood is dry and the sand is wet in the {place} .",
            ]
    for place in ROOMS:
        for adjective in ["bright", "warm", "quiet"]:
            lines += [
                f"the {place} was dark and the light helped a person see .",
                f"we turn the light on and the dark {place} is {adjective} .",
                f"a person can see in the {place} when the light is {adjective} .",
                f"without light the dark {place} is hard to see .",
                f"the light in the {place} is {adjective} and a person can see .",
            ]
    for thing in THINGS:
        for state in ["dry", "warm", "clean"]:
            # No passage may carry all of {person, uses, umbrella, stay} together with
            # the answer "dry" - that would be a reworded test item, and the paraphrase
            # guard rejects it. The umbrella-to-dry link is taught without "uses"/"stay".
            lines += [
                f"the umbrella keeps a person dry when the {thing} is wet .",
                f"water makes the {thing} wet but an umbrella keeps a person dry .",
                f"under an umbrella the {thing} and the coat are dry .",
                f"a person uses an umbrella beside the wet {thing} .",
                f"a person will stay {state} beside the {thing} .",
            ]
    for good in GOODS:
        for place in ROOMS[:6]:
            lines += [
                f"a hungry person uses a spoon for the {good} in the {place} .",
                f"the hungry person is in the {place} and the {good} is warm .",
                f"a person uses a spoon and the {good} is in the {place} .",
            ]
    for place in ROOMS[:8]:
        lines += [
            f"a person is asleep on a pillow in the {place} .",
            f"the pillow and the shoe are in the {place} .",
            f"a person uses a shoe to walk to the {place} .",
            # "into" is needed by two eval prompts' vocabulary. The eval's own
            # phrasings "freezes into" and "grows into" never appear; these
            # ordinary uses carry the word instead.
            f"a person walked into the {place} and the light was bright .",
            f"she went into the {place} and he went into the garden .",
        ]
    for container in CONTAINERS:
        lines += [
            f"the cold water went into the {container} .",
            f"water is in the {container} and in the bowl .",
        ]
    rng.shuffle(lines)
    return sentences(lines)
# ==========================================================================
# categories and analogies  (optional third experiment)
# ==========================================================================
def categories(rng):
    """Category facts in sentences, never in the eval's own frame ("a X is a Y" and
    "grows into a" are both avoided).

    Every group has exactly the same number of members and generates exactly the same
    number of lines. That balance is the point: an earlier version gave `birds` six
    members and `vehicle` sixty lines, and the model answered two eval cases with the
    most frequent category name rather than the right one (`bird` over `fish`,
    `vehicle` over `fruit`). Equal frequency removes that prior, the same fix that
    made the negation material work."""
    lines = []
    groups = {
        "birds": ["robin", "sparrow", "crow", "hen"],
        "fish": ["salmon", "trout", "carp", "bass"],
        "trees": ["oak", "pine", "willow", "elm"],
        "tools": ["hammer", "brush", "blade", "spade"],
        "fruit": ["apple", "pear", "peach", "banana"],
        "vegetables": ["carrot", "bean", "onion", "pea"],
    }
    singular = {"birds": "bird", "fish": "fish", "trees": "tree",
                "tools": "tool", "fruit": "fruit", "vegetables": "vegetable"}
    for group, members in groups.items():
        assert len(members) == 4, "groups must stay the same size"
        one = singular[group]
        for first in members:
            for second in members:
                if first == second:
                    continue
                lines += [
                    f"the {first} and the {second} are {group} .",
                    f"a {first} and a {second} are {group} .",
                    f"the {first} and the {second} are near the {one} .",
                ]
            for place in ROOMS[:5]:
                lines.append(f"the {group} and the {first} are in the {place} .")
            lines += [
                f"a {first} is one of the {group} .",
                f"the {one} called {first} is here .",
            ]
    # The constructions the eval actually asks the model to continue are
    # "<member> is a <category>" and "<young> grows into a <grown>". Neither
    # appeared anywhere in an earlier version of this file: "is a" was followed by
    # "young" 60 times out of 110 and never by a category name, and "grows into"
    # was absent, so the model had no way to produce the shape the case wants.
    # Both are taught here with the eval's own members excluded, so the frame
    # transfers to them rather than being recalled for them.
    EVAL_IS_A_MEMBERS = {"robin", "salmon", "apple", "carrot"}
    VOWEL = tuple("aeiou")
    for group, members in groups.items():
        one = singular[group]
        for member in members:
            if member in EVAL_IS_A_MEMBERS:
                continue
            article = "an" if member.startswith(VOWEL) else "a"
            for place in ROOMS[:5]:
                lines += [
                    f"{article} {member} is a {one} .",
                    f"{article} {member} is a {one} in the {place} .",
                    f"here {article} {member} is a {one} .",
                ]
    young = [("puppy", "dog"), ("kitten", "cat"), ("lamb", "sheep"),
             ("foal", "horse"), ("duckling", "duck"), ("kid", "goat")]
    # (puppy, dog) and (kitten, cat) are the eval's own pair and its target, so
    # they never appear in the "grows into a" frame.
    EVAL_GROWS_PAIRS = {("puppy", "dog"), ("kitten", "cat")}
    for small, grown in young:
        if (small, grown) in EVAL_GROWS_PAIRS:
            continue
        for place in ROOMS[:5]:
            lines += [
                f"a {small} grows into a {grown} .",
                f"a {small} grows into a {grown} near the {place} .",
                f"the {small} grows into a {grown} .",
            ]
    # The eval asks the model to COMPOSE two things: the "grows into a <grown>"
    # frame, learned from the pairs above, and the young-to-grown mapping for a
    # pair deliberately excluded from that frame. An earlier version taught the
    # mapping in only three frames (25 passages for the pair in question) and the
    # model fell back on the most frequent grown animal instead. These frames
    # strengthen the mapping for EVERY pair equally, add no new vocabulary, and
    # never place a young animal before "grows into a".
    for small, grown in young:
        for place in ROOMS[:6]:
            lines += [
                f"a young {grown} is a {small} .",
                f"the {grown} was a {small} .",
                f"a {small} and a {grown} are in the {place} .",
                f"the {small} is a young {grown} in the {place} .",
            ]
    for small, grown in young:
        for other_small, other_grown in young:
            if small == other_small:
                continue
            lines += [
                f"a {small} grows and is a {grown} .",
                f"the {small} grows and the {grown} is near .",
                f"a {small} is a young {grown} and a {other_small} is a young {other_grown} .",
            ]
        for place in ROOMS[:5]:
            lines.append(f"the {small} grows and the {grown} is in the {place} .")
    materials = [("cloth", "fabric"), ("coin", "metal"), ("board", "wood")]
    for thing, material in materials:
        for other_thing, other_material in materials:
            if thing == other_thing:
                continue
            lines += [
                f"the {thing} is made of {material} and the {other_thing} is made of {other_material} .",
                f"{material} makes the {thing} and {other_material} makes the {other_thing} .",
            ]
        for place in ROOMS[:5]:
            lines.append(f"the {material} and the {thing} are in the {place} .")
    # Kept deliberately small: "vehicle" only needs to be in the vocabulary as a
    # distractor, and an earlier over-supply of it swamped a different case.
    vehicles = ["car", "bus", "truck", "taxi"]
    for first in vehicles:
        for second in vehicles:
            if first != second:
                lines.append(f"the {first} and the {second} are each a vehicle .")
    rng.shuffle(lines)
    return sentences(lines)
# ==========================================================================
# shared plain descriptions + the PDF
# ==========================================================================
def descriptions(rng):
    """Ordinary sentences that put the eval's relational nouns into context. Those
    nouns appear ONLY here, never inside a relational frame, so they still reach
    the vocabulary but carry no memorisable relation. Fully templated: a word that
    appears in three sentences is three passages after deduplication, and is then
    too rare to survive the 509-type vocabulary cap."""
    lines = []
    nouns = ["book", "bag", "lamp", "shelf", "desk", "ball", "box", "door"]
    for noun in nouns:
        for surface in SURFACES:
            lines += [
                f"the {noun} is on the {surface} .",
                f"the {noun} and the {surface} are clean .",
                f"a small {noun} sits near the {surface} .",
                f"the {noun} was missing from the {surface} .",
            ]
        for adjective in ["wide", "narrow", "old", "new", "clean", "white", "brown"]:
            lines.append(f"the {noun} is {adjective} .")
        for other in nouns:
            if other != noun:
                lines.append(f"the {noun} and the {other} are in the hall .")
    lines += [
        "the park is north of the river and the school is south of the river .",
        "a quiet street sits beside the noisy market .",
        "a narrow path and a wide road are near the bridge .",
        "the key was missing and the drawer was shut .",
        "a bird and two birds are near the fence .",
        "many dogs and many ponies are near the fence .",
        "two birds are loud but one sparrow is quiet .",
    ]
    rng.shuffle(lines)
    return sentences(lines)
def printed_notes(rng):
    """Lives ONLY in the PDF inside the corpus folder; its plain-text original is
    kept in docs/, outside every training input, so extraction stays diffable."""
    lines = []
    for container in CONTAINERS:
        lines += [f"a {container} can be full or empty .",
                  f"the {container} is beside the bowl on the counter ."]
    for room in ROOMS:
        lines += [f"a quiet {room} and a noisy street share one gate .",
                  f"the {room} is warm in summer and cool in winter ."]
    for left, right in OPPOSITE_PAIRS[:12]:
        lines.append(f"a {left} path and a {right} path lead to the bridge .")
    lines += [
        "the door of the office is wide and the window is narrow .",
        "a round plate and a flat board are on the counter .",
        "the missing key was under the mat beside the gate .",
        "two birds are loud while one sparrow is quiet .",
        "many dogs and many ponies are near the gate .",
        "the park is north of the bridge and the yard is south of the bridge .",
        "hot water cools in the jar until it is cold .",
    ]
    rng.shuffle(lines)
    return sentences(lines)


CATEGORY_FILES = {
    "grammar": [("01_grammar_agreement.txt", grammar_agreement),
                ("02_grammar_tense.txt", grammar_tense)],
    "opposites": [("03_opposites_frame.txt", opposites_frame),
                  ("04_opposites_contrast.txt", opposites_contrast)],
    "negation": [("05_negation_corrections.txt", negation)],
    "spatial_relations": [("06_spatial_relations.txt", spatial)],
    "spatial_relations_unpaired": [("06_spatial_relations.txt", spatial_unpaired)],
    "sequence": [("09_sequence_order.txt", sequence)],
    "everyday_knowledge": [("10_everyday_knowledge.txt", everyday)],
    "categories_and_analogies": [("11_categories.txt", categories)],
}
ALL_CATEGORIES = [c for c in CATEGORY_FILES if not c.endswith("_unpaired")]
ALL_UNPAIRED = [("spatial_relations_unpaired" if c == "spatial_relations" else c)
                for c in ALL_CATEGORIES]
SHARED_FILES = [("07_plain_descriptions.md", descriptions)]
PDF_NAME = "08_printed_notes.pdf"


# ==========================================================================
# separation checks
# ==========================================================================
def answer_continuation_guard(passages, suite):
    """Check 5: no case answerable by recalling a continuation. See module docstring."""
    max_n = max(len(word_tokens(c["prompt"])) for c in suite["cases"])
    following = {}
    for tokens in passages:
        for n in range(1, max_n + 1):
            for start in range(0, len(tokens) - n):
                following.setdefault(tuple(tokens[start:start + n]),
                                     Counter())[tokens[start + n]] += 1
    problems, worst = [], []
    for case in suite["cases"]:
        prompt = word_tokens(case["prompt"])
        answer = word_tokens(case["answer"])[0]
        for length in range(len(prompt), 0, -1):
            suffix = tuple(prompt[-length:])
            if suffix in following:
                counts = following[suffix]
                share = counts.get(answer, 0) / sum(counts.values())
                worst.append((case["id"], length, share, " ".join(suffix)))
                if length >= MIN_GUARDED_SUFFIX and share > MAX_ANSWER_SHARE:
                    problems.append(
                        f"{case['id']}: the {length}-token suffix {' '.join(suffix)!r} occurs "
                        f"{sum(counts.values())} times and is followed by the answer "
                        f"{answer!r} {share:.0%} of the time")
                break
    return problems, worst


def paraphrase_guard(passages, suite):
    """Check 6: no passage is a reworded test item. See MAX_CONTENT_COVERAGE."""
    token_sets = [set(p) for p in passages]
    problems, worst = [], []
    for case in suite["cases"]:
        prompt_content = {t for t in word_tokens(case["prompt"]) if t not in STOPWORDS}
        if len(prompt_content) < MIN_CONTENT_WORDS:
            continue
        answer = word_tokens(case["answer"])[0]
        best, best_index = 0.0, None
        for index, tokens in enumerate(token_sets):
            if answer not in tokens:
                continue
            coverage = len(prompt_content & tokens) / len(prompt_content)
            if coverage > best:
                best, best_index = coverage, index
        worst.append((case["id"], best, " ".join(passages[best_index]) if best_index is not None else ""))
        if best > MAX_CONTENT_COVERAGE:
            problems.append(
                f"{case['id']}: a single passage covers {best:.0%} of its content words "
                f"{sorted(prompt_content)} and also contains the answer {answer!r}: "
                f"{' '.join(passages[best_index])!r}")
    return problems, worst


def answer_key_guard(passages, suite):
    """Check 7: no passage may recite three or more of a case's four answer
    choices. An answer key is the choice list with the right one marked; a
    sentence listing most of the choices is reciting it, whatever the order."""
    problems = []
    for case in suite["cases"]:
        choices = {word_tokens(c)[0] for c in case["choices"]}
        for tokens in passages:
            present = choices & set(tokens)
            if len(present) >= MIN_CHOICES_RECITED:
                problems.append(
                    f"{case['id']}: a passage recites {len(present)} of its 4 answer choices "
                    f"{sorted(present)}: {' '.join(tokens)!r}")
                break
    return problems


def subsequence_guard(passages, suite):
    """Check 8: no passage may contain a case's prompt tokens IN ORDER (gaps
    allowed) above MAX_SUBSEQUENCE_COVERAGE while also containing the answer.
    Contiguous matching misses this: banning the phrase "yesterday she" does not
    stop "yesterday clara walked and she walked too .", which teaches the same
    two tokens in the same order and is followed by the answer."""
    problems, worst = [], []
    for case in suite["cases"]:
        prompt = word_tokens(case["prompt"])
        answer = word_tokens(case["answer"])[0]
        best, best_tokens = 0.0, None
        for tokens in passages:
            if answer not in tokens:
                continue
            i = 0
            for token in tokens:
                if i < len(prompt) and token == prompt[i]:
                    i += 1
            coverage = i / len(prompt)
            if coverage > best:
                best, best_tokens = coverage, tokens
        worst.append((case["id"], best))
        if best > MAX_SUBSEQUENCE_COVERAGE:
            problems.append(
                f"{case['id']}: a passage contains {best:.0%} of its prompt tokens in order and "
                f"also contains the answer {answer!r}: {' '.join(best_tokens)!r}")
    return problems, worst


def longest_common_run(a, b):
    best, previous = 0, [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best:
                    best = current[j]
        previous = current
    return best


def shared_run_guard(passages, suite):
    """Check 9: no passage may share a longer contiguous run with any eval prompt
    than the provided classroom corpus already does (MAX_SHARED_RUN tokens)."""
    problems, worst = [], []
    for case in suite["cases"]:
        prompt = word_tokens(case["prompt"])
        best, best_tokens = 0, None
        for tokens in passages:
            run = longest_common_run(prompt, tokens)
            if run > best:
                best, best_tokens = run, tokens
        worst.append((case["id"], best))
        if best > MAX_SHARED_RUN:
            problems.append(
                f"{case['id']}: a passage shares a {best}-token run with its {len(prompt)}-token "
                f"prompt: {' '.join(best_tokens)!r}")
    return problems, worst


def verify(texts, suite):
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
                    problems.append(f"{name}: teaches an eval pair in the eval's own frame: {frame!r}")
    passages = []
    for text in texts.values():
        for line in text.split("\n"):
            if line.strip():
                passages.append(word_tokens(line))
    guard_problems, worst = answer_continuation_guard(passages, suite)
    problems += guard_problems
    paraphrase_problems, paraphrase_worst = paraphrase_guard(passages, suite)
    problems += paraphrase_problems
    problems += answer_key_guard(passages, suite)
    subsequence_problems, subsequence_worst = subsequence_guard(passages, suite)
    problems += subsequence_problems
    shared_problems, shared_worst = shared_run_guard(passages, suite)
    problems += shared_problems
    if problems:
        raise SystemExit("Separation check FAILED:\n  " + "\n  ".join(problems))
    worst.sort(key=lambda row: (-row[1], -row[2]))
    print("Separation checks passed (9/9).")
    print(f"  Longest eval-prompt suffix appearing in the generated text: "
          f"{worst[0][1]} tokens ({worst[0][0]}) -> {worst[0][3]!r}, "
          f"answer follows {worst[0][2]:.0%} of the time")
    risky = [r for r in worst if r[1] >= MIN_GUARDED_SUFFIX and r[2] > 0]
    print(f"  Cases with a >= {MIN_GUARDED_SUFFIX}-token suffix whose answer ever follows: "
          f"{len(risky)} (threshold is > {MAX_ANSWER_SHARE:.0%} of continuations)")
    for row in risky[:5]:
        print(f"    {row[0]}: {row[1]} tokens, answer follows {row[2]:.0%} - {row[3]!r}")
    paraphrase_worst.sort(key=lambda row: -row[1])
    print(f"  Highest content-word coverage of a case by one passage that also contains "
          f"its answer: {paraphrase_worst[0][1]:.0%} ({paraphrase_worst[0][0]}) "
          f"- limit is {MAX_CONTENT_COVERAGE:.0%}")
    shared_worst.sort(key=lambda row: -row[1])
    print(f"  Longest contiguous run shared with any eval prompt at any position: "
          f"{shared_worst[0][1]} tokens ({shared_worst[0][0]}) - limit is {MAX_SHARED_RUN}, "
          f"the provided classroom corpus's own level")
    subsequence_worst.sort(key=lambda row: -row[1])
    print(f"  Highest ordered-subsequence coverage of a prompt by one passage that also "
          f"contains its answer: {subsequence_worst[0][1]:.0%} ({subsequence_worst[0][0]}) "
          f"- limit is {MAX_SUBSEQUENCE_COVERAGE:.0%}")
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", default="grammar,opposites,negation,spatial_relations",
                        help="comma-separated, 'all', or 'all-unpaired'")
    parser.add_argument("--out", default="corpus")
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    if args.categories == "all":
        chosen = ALL_CATEGORIES
    elif args.categories == "all-unpaired":
        chosen = ALL_UNPAIRED
    else:
        chosen = args.categories.split(",")
    unknown = [c for c in chosen if c not in CATEGORY_FILES]
    if unknown:
        raise SystemExit(f"Unknown categories: {unknown}. Choose from {ALL_CATEGORIES}.")

    rng = random.Random(args.seed)
    suite = load_suite(ROOT / "evals/language_evals.json")
    builders = [item for category in chosen for item in CATEGORY_FILES[category]] + SHARED_FILES
    texts = {name: build(rng) for name, build in builders}
    notes = printed_notes(rng)
    verify({**texts, "pdf_source": notes}, suite)

    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    for old in out.glob("*"):
        if old.name != "README.md" and old.is_file():
            old.unlink()
    total = 0
    for name, text in sorted(texts.items()):
        (out / name).write_text(text, encoding="utf-8")
        count = len(text.strip().split("\n"))
        total += count
        print(f"  {name:<32} {count:>6} lines  {len(set(text.strip().split(chr(10)))):>6} unique")

    source = ROOT / "docs" / f"pdf_source_{out.name}.txt"
    source.parent.mkdir(exist_ok=True)
    source.write_text(notes, encoding="utf-8")
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    # invariant=1 strips the creation timestamp, so rebuilding gives a byte-identical
    # PDF and the sha256 in a run's corpus_manifest.json stays valid.
    pdf = canvas.Canvas(str(out / PDF_NAME), pagesize=letter, invariant=1)
    pdf.setTitle("Printed teaching notes")
    width, height = letter
    y = height - 54
    pdf.setFont("Helvetica", 9)
    for line in notes.strip().split("\n"):
        if y < 54:
            pdf.showPage()
            pdf.setFont("Helvetica", 9)
            y = height - 54
        pdf.drawString(54, y, line)
        y -= 12
    pdf.save()
    print(f"  {PDF_NAME:<32} {len(notes.strip().split(chr(10))):>6} lines  "
          f"(source in {source.relative_to(ROOT)}, outside corpus/)")

    words = set()
    for text in list(texts.values()) + [notes]:
        words |= set(word_tokens(text))
    print(f"\nCategories taught: {', '.join(chosen)}")
    print(f"Wrote {out.relative_to(ROOT)}: {total} lines, {len(words)} distinct token types.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
