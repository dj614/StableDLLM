import re
from typing import List, Tuple

# -- MD config --
MD_RE = re.compile(r'^[A-Z]+[0-9]+$')

# -- HTML config --
HTML_PREFIX = (
    "<td", "</td", "<th", "</th", "<tr", "</tr",
    "<table", "</table", "<thead", "</thead", "<tbody", "</tbody"
)

# -- TEG-DB config --
TEG_JSON_SYMS = {"{", "}", "[", "]", ":", ",", '"', "'"}

# -- CodeXGLUE config --
CODEX_STRUCTURED = {
    # Python keywords: 15
    'as','class','def','else','except','finally','for','from',
    'if','import','raise','return','try','while','with',
    # single‐char punctuation/operators: 18
    '(',')','{','}','[',']',':',',','.', '+','-','*','/','<','=','>','#','@',
    # multi‐char operators: 12
    '->','!=','==','>=','<=','+=','-=','*=','/=','::','...','..',
    # quotes: 4
    '"',"'",'"""',"'''"
}


def is_coord_token(tok: str, fmt: str) -> bool:
    """
    判断 token tok（去除 'Ġ' 前缀后）是否为结构化符号。
    对 codexglue-json 分支仅做硬过滤：tok_clean 必须完全落在 CODEX_STRUCTURED 中。
    """
    tok_clean = tok.lstrip('Ġ')

    if fmt == 'hitab-markdown':
        return bool(MD_RE.fullmatch(tok_clean))

    elif fmt == 'hitab-html':
        low = tok_clean.lower()
        return (
            any(low.startswith(p) for p in HTML_PREFIX)
            or low in {"<", ">", "</", "><", "></", "td", "th", "tr", "table", "thead", "tbody"}
        )

    elif fmt == 'tegdb-json':
        return tok_clean in TEG_JSON_SYMS

    elif fmt == 'codexglue-json':
        if tok_clean in CODEX_STRUCTURED:
            return True
        if tok == 'Ċ':
            return True
        if set(tok) == {'Ġ'} and len(tok) >= 2:
            return True
        return False

    else:
        return False
