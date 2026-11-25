"""Finnish front end for tiny-voca: text -> phoneme tokens with a stress feature."""
from dataclasses import dataclass
import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata

FRONTEND_VERSION = 301
ESPEAK_VOICE = "fi"
SEPARATOR = "_"
FRONTENDS = ("espeak", "own")
FRONTEND_VARIABLE = "TINY_VOCA_FRONTEND"
_frontend = os.environ.get(FRONTEND_VARIABLE, "espeak")

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")
ORIG_CONSONANTS = (
    "p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    "m", "n", "ŋ", "l", "ɹ", "j", "w", "tʃ", "dʒ", "ɾ", "ʔ",
)
ORIG_VOWELS = (
    "ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ", "uː", "ɜː",
    "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "iə", "oː",
    "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl", "n̩",
)
FI_NEW_CONSONANTS = ("r", "pː", "tː", "kː")
FI_NEW_VOWELS = ("a", "aː", "e", "eː", "o", "u", "y", "yː", "æː", "ø", "øː", "ɪː")
CONSONANTS = ORIG_CONSONANTS + FI_NEW_CONSONANTS
VOWELS = ORIG_VOWELS + FI_NEW_VOWELS
PHONES = ORIG_CONSONANTS + ORIG_VOWELS + FI_NEW_CONSONANTS + FI_NEW_VOWELS
SYMBOLS = SPECIALS + PUNCTUATION + PHONES
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_SET = frozenset(PHONES)
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES) | {SYMBOL_TO_ID[UNK]}
VOICELESS = frozenset(("p", "pː", "t", "tː", "k", "kː", "f", "θ", "s", "ʃ", "h", "tʃ", "ʔ"))
VOWEL_SET = frozenset(VOWELS)
STRESS_MARKS = {"ˈ": 1, "ˌ": 2}
FALLBACK = {
    "ai": ("a", "i"), "ei": ("e", "i"), "oi": ("o", "i"), "ui": ("u", "i"), "yi": ("y", "i"),
    "æi": ("æ", "i"), "øi": ("ø", "i"), "au": ("a", "u"), "eu": ("e", "u"), "iu": ("i", "u"),
    "ou": ("o", "u"), "ey": ("e", "y"), "iy": ("i", "y"), "æy": ("æ", "y"), "øy": ("ø", "y"),
    "ie": ("i", "e"), "uo": ("u", "o"), "yø": ("y", "ø"),
    "ɑ": ("a",), "ɒ": ("o",), "ɑ̃": ("aː",), "ɔ̃": ("oː",), "ɛ̃": ("eː",),
    "œ": ("ø",), "œː": ("øː",), "ɜ": ("ø",), "ɛː": ("eː",),
    "ʉ": ("y",), "ʉː": ("yː",), "ʏ": ("y",),
    "bː": ("b",), "dː": ("d",), "ɡː": ("ɡ",), "g": ("ɡ",), "gː": ("ɡ",),
    "sː": ("s", "s"), "lː": ("l", "l"), "mː": ("m", "m"), "nː": ("n", "n"), "rː": ("r", "r"), "ŋː": ("ŋ", "ŋ"),
    "fː": ("f", "f"), "vː": ("v", "v"), "hː": ("h", "h"), "jː": ("j", "j"),
    "q": ("k",), "x": ("k",), "ç": ("h",), "β": ("v",), "ʋ": ("v",), "ɣ": ("ɡ",),
    "ɻ": ("r",), "ʁ": ("r",), "ɬ": ("l",), "ɫ": ("l",), "ɲ": ("n", "j"), "ʎ": ("l", "j"),
    "ts": ("t", "s"), "dz": ("d", "z"), "tʃː": ("tː", "ʃ"),
    "əʊ": ("o", "u"), "eə": ("e", "a"), "ɪə": ("i", "a"), "ʊə": ("u", "a"),
    "l̩": ("l",), "m̩": ("m",), "n̩": ("n",),
}
_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
    "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")",
})
_PUNCTUATION_CHARS = frozenset(PUNCTUATION)
ABBREVIATIONS = {"esim", "mm", "jne", "yms", "ym", "ns", "vrt", "ks", "klo", "n", "v", "s", "t", "k", "r", "dr", "prof", "mr", "mrs", "ms"}
_LANGUAGE_TAG = re.compile(r"\((?:_?[a-z0-9]_?){2,3}(?:-(?:_?[a-z0-9]_?)+)*\)")


class NothingToPronounce(ValueError):
    pass


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple

    def __len__(self):
        return len(self.ids)


def frontend():
    return _frontend


def set_frontend(name):
    global _frontend
    _frontend = name


def frontend_identity():
    return {
        "frontend_version": FRONTEND_VERSION,
        "voice": ESPEAK_VOICE,
        "separator": SEPARATOR,
        "symbols": list(SYMBOLS),
        "fallback": {key: list(value) for key, value in sorted(FALLBACK.items())},
    }


def frontend_runtime():
    return {"espeak_version": espeak_version()}


def frontend_fingerprint():
    return hashlib.sha256(json.dumps(frontend_identity(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"-{2,}|\s-\s", "—", text)
    return re.sub(r"\s+", " ", text).strip()


def chunks(text):
    start = 0
    for i, char in enumerate(text):
        if char not in _PUNCTUATION_CHARS:
            continue
        previous = text[i - 1] if i else ""
        following = text[i + 1] if i + 1 < len(text) else ""
        if char in "':." and previous.isalnum() and following.isalnum():
            continue
        if char in ".,:" and previous.isdigit() and following.isdigit():
            continue
        if char == ".":
            word = re.search(r"([A-Za-zÄÖÅäöå]+)$", text[:i])
            if word and word[1].lower() in ABBREVIATIONS:
                continue
        if text[start:i].strip():
            yield False, text[start:i].strip()
        yield True, char
        start = i + 1
    if text[start:].strip():
        yield False, text[start:].strip()


def espeak_executable():
    return shutil.which("espeak-ng") or r"C:\Program Files\eSpeak NG\espeak-ng.exe"


def espeak_version():
    return subprocess.run([espeak_executable(), "--version"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()


def espeak_words(piece):
    result = subprocess.run(
        [espeak_executable(), "-q", "--ipa", "--sep=" + SEPARATOR, "-v", ESPEAK_VOICE, "--stdin"],
        input=piece.rstrip("\n") + "\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=30,
    )
    return parse_espeak(result.stdout)


def parse_espeak(output):
    words = []
    for word in _LANGUAGE_TAG.sub("", output).split():
        units = []
        for unit in word.split(SEPARATOR):
            stress = 0
            for mark, level in STRESS_MARKS.items():
                if mark in unit:
                    stress = stress or level
                    unit = unit.replace(mark, "")
            unit = "".join(c for c in unit if unicodedata.category(c) != "Cf" and c not in "()ˈˌ")
            if unit:
                units.append((unit, stress))
        if units:
            words.append(units)
    return words


def canonical(phone):
    if phone in PHONE_SET:
        return (phone,)
    if phone in FALLBACK:
        return FALLBACK[phone]
    stripped = "".join(c for c in unicodedata.normalize("NFD", phone) if not unicodedata.combining(c))
    for candidate in (stripped, stripped.replace("ː", ""), stripped + "ː"):
        if candidate != phone and candidate in PHONE_SET:
            return (candidate,)
        if candidate != phone and candidate in FALLBACK:
            return FALLBACK[candidate]
    return None


def tokenize(pieces):
    symbols, stress, unknown = [BOS], [0], []
    for punctuation, value in pieces:
        if punctuation:
            if symbols[-1] != value:
                symbols.append(value)
                stress.append(0)
            continue
        for word in value:
            if symbols[-1] in PHONE_SET or symbols[-1] == UNK:
                symbols.append(SP)
                stress.append(0)
            for phone, level in word:
                mapped = canonical(phone)
                if mapped is None:
                    unknown.append(phone)
                    mapped = (UNK,)
                carrier = next((i for i, item in enumerate(mapped) if item in VOWEL_SET), 0)
                for index, item in enumerate(mapped):
                    symbols.append(item)
                    stress.append(level if index == carrier else 0)
    symbols.append(EOS)
    stress.append(0)
    if not any(s in PHONE_SET or s == UNK for s in symbols):
        raise NothingToPronounce("nothing")
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), tuple(stress), tuple(unknown))


def phonemize(text):
    pieces = [(True, piece) if punctuation else (False, espeak_words(piece)) for punctuation, piece in chunks(normalize_text(text))]
    return tokenize(pieces)


def is_phone(token_ids):
    return [int(i) in PHONE_IDS for i in token_ids]
