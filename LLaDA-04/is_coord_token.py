import re

# -- MD config --
MD_RE = re.compile(r'^[A-Z]+[0-9]+$')

# -- HTML config --
HTML_PREFIX = (
    "<td", "</td", "<th", "</th", "<tr", "</tr",
    "<table", "</table", "<thead", "</thead", "<tbody", "</tbody"
)

# -- cora_pubmed config --
_STRUCT_PHRASES = {
    "ĠTitle", "Abstract", "-hop", "Ġneighbor", "Ġtext", "Ġinformation", ":"
}

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
    '"',"'",'"""',"'''",
    # new line: 1
    'Ċ'
}
# CODEX_STRUCTURED = {
#     "#", "else", "class", "==", "try", "while", "<", "except", "*", "if",
#     "for", "with", "!=", "'''", "raise", "return", "def", "from", '"',
#     '"""', "::", "as", "..", "/", "Ċ", "import", "{", "...", "}", "'", "+", "@", "-", ",", "[", "(", ":", '"', ")", "=", "]", ">"
# }

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

    elif fmt == 'cora_pubmed':
        low = tok_clean.lower()
        # 若 token 本身完整等于其中一句 → True
        if low in _STRUCT_PHRASES:
            return True
        return False

    elif fmt == 'codexglue-json':
        if tok_clean in CODEX_STRUCTURED:
            return True
        if set(tok) == {'Ġ'} and len(tok) in (3, 7, 11, 15, 19):
            return True
        return False

    else:
        return False
