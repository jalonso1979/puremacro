"""Stop-word lists for the languages the narrative connectors return.

English and Spanish are the lists :mod:`puremacro.narrative.topics` has always
used (unchanged, so existing topic models keep their vocabularies). Portuguese,
German, French and Italian are short function-word lists: closed-class words
(articles, pronouns, prepositions, conjunctions, auxiliaries) only, so no
economically meaningful word is dropped by default.
"""
from __future__ import annotations

from typing import Collection

EN = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "also", "may", "said", "one", "two", "well", "will"
})

ES = frozenset({
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un",
    "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le",
    "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre",
    "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante",
    "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
    "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro", "otras",
    "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual",
    "sea", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis",
    "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros", "vosotras", "os",
    "mío", "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya",
    "suyos", "suyas", "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra",
    "vuestros", "vuestras", "esos", "esas", "estoy", "estás", "está", "estamos",
    "estáis", "están", "esté", "estés", "estemos", "estéis", "estén", "asimismo", "dicho"
})

PT = frozenset({
    "a", "à", "às", "ao", "aos", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "num", "numa",
    "por", "pelo", "pela", "pelos", "pelas", "para", "com", "sem", "sob", "sobre",
    "entre", "até", "desde", "após", "e", "ou", "mas", "que", "se", "como",
    "quando", "onde", "porque", "pois", "também", "já", "não", "mais", "muito",
    "eu", "tu", "ele", "ela", "nós", "vós", "eles", "elas", "me", "te", "lhe",
    "lhes", "seu", "sua", "seus", "suas", "meu", "minha", "nosso", "nossa",
    "este", "esta", "estes", "estas", "isto", "esse", "essa", "esses", "essas",
    "isso", "aquele", "aquela", "aquilo", "ser", "é", "são", "foi", "foram",
    "era", "estar", "está", "estão", "esteve", "ter", "tem", "têm", "teve",
    "há", "havia", "seja", "sido", "pelo", "qual", "quais",
})

DE = frozenset({
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines",
    "einem", "einen", "und", "oder", "aber", "doch", "sondern", "denn", "dass",
    "daß", "wenn", "als", "wie", "ob", "weil", "da", "in", "im", "ins", "an",
    "am", "auf", "aus", "bei", "beim", "mit", "nach", "von", "vom", "zu", "zum",
    "zur", "für", "über", "unter", "vor", "durch", "gegen", "ohne", "um", "bis",
    "seit", "zwischen", "ich", "du", "er", "sie", "es", "wir", "ihr", "ihm",
    "ihn", "ihnen", "sein", "seine", "seiner", "seinem", "seinen", "ihre",
    "ihrer", "ihrem", "ihren", "unser", "unsere", "dieser", "diese", "dieses",
    "diesem", "diesen", "jener", "jene", "ist", "sind", "war", "waren", "wird",
    "werden", "wurde", "wurden", "worden", "hat", "haben", "hatte", "hatten",
    "kann", "können", "soll", "sollen", "nicht", "auch", "noch", "nur", "so",
    "sich", "man", "was", "wer", "welche", "welcher", "welches",
})

FR = frozenset({
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "au", "aux",
    "et", "ou", "mais", "donc", "ni", "car", "que", "qu", "qui", "quoi",
    "dont", "où", "si", "comme", "quand", "à", "en", "dans", "par", "pour",
    "sur", "sous", "avec", "sans", "entre", "vers", "chez", "depuis", "pendant",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "me", "te",
    "se", "lui", "leur", "leurs", "son", "sa", "ses", "mon", "ma", "mes", "ton",
    "ta", "tes", "notre", "nos", "votre", "vos", "ce", "cet", "cette", "ces",
    "ceci", "cela", "ça", "est", "sont", "été", "être", "était", "étaient",
    "sera", "seront", "a", "ont", "avait", "avaient", "avoir", "aura", "ne",
    "pas", "plus", "aussi", "très", "y", "c", "s", "n", "j", "m", "t",
})

IT = frozenset({
    "il", "lo", "la", "i", "gli", "le", "l", "un", "uno", "una", "un'", "di",
    "del", "dello", "della", "dei", "degli", "delle", "a", "al", "allo", "alla",
    "ai", "agli", "alle", "da", "dal", "dallo", "dalla", "dai", "dagli", "dalle",
    "in", "nel", "nello", "nella", "nei", "negli", "nelle", "con", "su", "sul",
    "sullo", "sulla", "sui", "sugli", "sulle", "per", "tra", "fra", "e", "ed",
    "o", "ma", "che", "se", "come", "quando", "dove", "perché", "anche", "non",
    "più", "io", "tu", "lui", "lei", "noi", "voi", "loro", "mi", "ti", "ci",
    "vi", "si", "ne", "suo", "sua", "suoi", "sue", "mio", "mia", "nostro",
    "nostra", "questo", "questa", "questi", "queste", "quello", "quella",
    "quelli", "quelle", "è", "sono", "era", "erano", "essere", "stato", "stata",
    "ha", "hanno", "aveva", "avere", "sia", "cui", "quale", "quali",
})

STOPWORDS: dict[str, frozenset[str]] = {
    "en": EN, "es": ES, "pt": PT, "de": DE, "fr": FR, "it": IT,
}

_ALIASES = {
    "english": "en", "spanish": "es", "español": "es", "portuguese": "pt",
    "português": "pt", "german": "de", "deutsch": "de", "french": "fr",
    "français": "fr", "italian": "it", "italiano": "it",
}


def normalize_language(language: str | None) -> str | None:
    """Map ``"English"``, ``"en-US"``, ``"pt_BR"`` and similar to a two-letter code."""
    if not language:
        return None
    lang = str(language).strip().lower()
    lang = _ALIASES.get(lang, lang)
    return lang.replace("_", "-").split("-")[0] or None


def stopwords(language: str | None) -> frozenset[str]:
    """Stop words for ``language``; an empty set for a language with no list."""
    return STOPWORDS.get(normalize_language(language) or "", frozenset())


def resolve_stop_words(
    spec: str | Collection[str] | None,
) -> frozenset[str] | None:
    """Turn a ``stop_words`` argument into a fixed set, or ``None`` for "per document".

    ``"auto"`` returns ``None``: the caller picks the list from each document's
    language. A language name or code returns that list; a collection is used
    as given; ``None`` disables stop-word removal.
    """
    if spec is None:
        return frozenset()
    if isinstance(spec, str):
        if spec.lower() == "auto":
            return None
        code = normalize_language(spec)
        if code not in STOPWORDS:
            raise ValueError(
                f"no stop-word list for {spec!r}; available: {sorted(STOPWORDS)}. "
                "Pass a set of words instead."
            )
        return STOPWORDS[code]
    return frozenset(spec)


__all__ = ["STOPWORDS", "stopwords", "normalize_language", "resolve_stop_words"]
