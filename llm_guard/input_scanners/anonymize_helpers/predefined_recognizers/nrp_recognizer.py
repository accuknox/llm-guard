from typing import List, Optional

from presidio_analyzer import PatternRecognizer


class NrpRecognizer(PatternRecognizer):
    """Recognizes religious and political group terms as NRP.

    The spaCy model behind the NRP SpacyRecognizer misses common terms depending on
    casing and context (e.g. it never tags lowercase "hindu"). This deny list makes the
    listed terms match regardless of case. Words that are also everyday adjectives
    ("liberal", "conservative", "green", "orthodox") and bare "Congress" are left out
    to avoid redacting ordinary prompts.
    """

    RELIGIONS = [
        "Hindu",
        "Hindus",
        "Hinduism",
        "Muslim",
        "Muslims",
        "Islam",
        "Islamic",
        "Christian",
        "Christians",
        "Christianity",
        "Catholic",
        "Catholics",
        "Catholicism",
        "Protestant",
        "Protestants",
        "Evangelical",
        "Evangelicals",
        "Mormon",
        "Mormons",
        "Sikh",
        "Sikhs",
        "Sikhism",
        "Jain",
        "Jains",
        "Jainism",
        "Buddhist",
        "Buddhists",
        "Buddhism",
        "Jew",
        "Jews",
        "Jewish",
        "Judaism",
        "Parsi",
        "Parsis",
        "Zoroastrian",
        "Zoroastrians",
        "Zoroastrianism",
        "Baha'i",
        "Bahai",
        "Sunni",
        "Sunnis",
        "Shia",
        "Shias",
        "Shiite",
        "Shiites",
        "Taoist",
        "Taoists",
        "Taoism",
        "Shinto",
        "Atheist",
        "Atheists",
        "Atheism",
        "Agnostic",
        "Agnostics",
    ]

    POLITICAL_GROUPS = [
        "Democrat",
        "Democrats",
        "Republican",
        "Republicans",
        "Communist",
        "Communists",
        "Socialist",
        "Socialists",
        "Marxist",
        "Marxists",
        "Maoist",
        "Maoists",
        "Fascist",
        "Fascists",
        "Nazi",
        "Nazis",
        "Libertarian",
        "Libertarians",
        "Tory",
        "Tories",
        "Democratic Party",
        "Republican Party",
        "Labour Party",
        "Conservative Party",
        "Liberal Democrats",
        "BJP",
        "Bharatiya Janata Party",
        "Indian National Congress",
        "Aam Aadmi Party",
    ]

    def __init__(
        self,
        deny_list: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "NRP",
        deny_list_score: float = 0.85,
    ):
        terms = deny_list or self.RELIGIONS + self.POLITICAL_GROUPS
        super().__init__(
            supported_entity=supported_entity,
            # Longest first, so "Republican Party" wins over "Republican".
            deny_list=sorted(terms, key=len, reverse=True),
            deny_list_score=deny_list_score,
            supported_language=supported_language,
        )
