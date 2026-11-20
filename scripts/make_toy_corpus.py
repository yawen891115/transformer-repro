"""Generate a tiny German->English corpus packed exactly like the real IWSLT'14 archive.

The archive (de-en.tgz) contains train.tags.de-en.{de,en} with TED-style tag lines and
IWSLT14.TED*.{de,en}.xml dev/test files, so both data-preparation scripts, preprocess.py,
train.py and generate.py can be tested end to end in minutes on a CPU. The sentences come
from a small grammar with real German word order (verb-second, participles and verbs at the
end of clauses), so the model has to learn reordering, not just a word-by-word lookup.

    python scripts/make_toy_corpus.py --out data/toy/orig/de-en.tgz
"""

from __future__ import annotations

import argparse
import io
import random
import tarfile
from pathlib import Path

NOUNS = [  # (German, gender, English)
    ("Hund", "m", "dog"), ("Mann", "m", "man"), ("Vogel", "m", "bird"), ("Apfel", "m", "apple"),
    ("Lehrer", "m", "teacher"), ("Ball", "m", "ball"), ("Katze", "f", "cat"),
    ("Frau", "f", "woman"), ("Blume", "f", "flower"), ("Stadt", "f", "city"),
    ("Tür", "f", "door"), ("Kind", "n", "child"), ("Haus", "n", "house"), ("Buch", "n", "book"),
    ("Auto", "n", "car"), ("Mädchen", "n", "girl"),
]
ADJS = [("groß", "big"), ("klein", "small"), ("rot", "red"), ("alt", "old"), ("jung", "young"),
        ("schnell", "fast"), ("schön", "beautiful"), ("neu", "new")]
VERBS = [  # (present 3sg, participle, English present, English participle)
    ("sieht", "gesehen", "sees", "seen"), ("findet", "gefunden", "finds", "found"),
    ("kauft", "gekauft", "buys", "bought"), ("sucht", "gesucht", "looks for", "looked for"),
    ("liebt", "geliebt", "loves", "loved"), ("malt", "gemalt", "paints", "painted"),
]
ART = {("nom", "m"): "der", ("nom", "f"): "die", ("nom", "n"): "das",
       ("acc", "m"): "den", ("acc", "f"): "die", ("acc", "n"): "das"}


def noun_phrase(rng: random.Random, case: str):
    de, gender, en = rng.choice(NOUNS)
    art = ART[(case, gender)]
    if rng.random() < 0.5:
        adj_de, adj_en = rng.choice(ADJS)
        ending = "en" if (case, gender) == ("acc", "m") else "e"   # weak adjective endings
        return f"{art} {adj_de}{ending} {de}", f"the {adj_en} {en}"
    return f"{art} {de}", f"the {en}"


def sentence(rng: random.Random):
    s_de, s_en = noun_phrase(rng, "nom")
    o_de, o_en = noun_phrase(rng, "acc")
    v_de, part_de, v_en, part_en = rng.choice(VERBS)
    kind = rng.random()
    if kind < 0.4:     # present tense, verb second
        de, en = f"{s_de} {v_de} {o_de}.", f"{s_en} {v_en} {o_en}."
    elif kind < 0.75:  # perfect tense: the participle goes to the end in German
        de, en = f"{s_de} hat {o_de} {part_de}.", f"{s_en} has {part_en} {o_en}."
    elif kind < 0.95:  # subordinate clause: the verb goes to the end in German
        de, en = f"Ich glaube, dass {s_de} {o_de} {v_de}.", f"I think that {s_en} {v_en} {o_en}."
    else:              # an apostrophe, to exercise the ’ -> ' replacement of the dev/test files
        de, en = "Das stimmt.", "It’s true."
    return de[0].upper() + de[1:], en[0].upper() + en[1:]


def train_file(rng: random.Random, n_talks: int, per_talk: int):
    de, en = [], []
    for talk in range(n_talks):
        for tag in (f"<url>http://www.ted.com/talks/toy_{talk}.html</url>",
                    "<keywords>talks, toys</keywords>", f"<talkid>{talk}</talkid>"):
            de.append(tag)
            en.append(tag)
        de.append(f"<title>Spielzeug Vortrag {talk}</title>")
        en.append(f"<title>Toy talk {talk}</title>")
        de.append("<description>Ein kleiner Vortrag.</description>")
        en.append("<description>A small talk.</description>")
        for _ in range(per_talk):
            d, e = sentence(rng)
            de.append(d)
            en.append(e.replace("’", "'"))   # the training side uses plain apostrophes
    return "\n".join(de) + "\n", "\n".join(en) + "\n"


def xml_file(rng: random.Random, setid: str, lang: str, pairs) -> str:
    segs = "\n".join(f'<seg id="{i + 1}"> {(p[0] if lang == "de" else p[1])} </seg>'
                     for i, p in enumerate(pairs))
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<mteval>\n<srcset setid="{setid}" '
            f'srclang="{lang}">\n<doc docid="1" genre="lectures">\n<talkid>1</talkid>\n'
            f"<title>Toy</title>\n{segs}\n</doc>\n</srcset>\n</mteval>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/toy/orig/de-en.tgz")
    parser.add_argument("--talks", type=int, default=120)
    parser.add_argument("--per-talk", type=int, default=100)
    parser.add_argument("--test-per-set", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    files = {}
    files["de-en/train.tags.de-en.de"], files["de-en/train.tags.de-en.en"] = \
        train_file(rng, args.talks, args.per_talk)
    for setid in ("IWSLT14.TED.dev2010", "IWSLT14.TEDX.dev2012", "IWSLT14.TED.tst2010",
                  "IWSLT14.TED.tst2011", "IWSLT14.TED.tst2012"):
        pairs = [sentence(rng) for _ in range(args.test_per_set)]
        for lang in ("de", "en"):
            files[f"de-en/{setid}.de-en.{lang}.xml"] = xml_file(rng, setid, lang, pairs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    print(f"wrote {out}: {args.talks * args.per_talk} training sentences, "
          f"{5 * args.test_per_set} dev/test sentences")


if __name__ == "__main__":
    main()
