import os
from typing import List, Dict


def load_biographies(path: str) -> List[Dict]:
    """
    Load biographies.md and chunk by page (using '---' as delimiter).
    Returns a list of dicts: [{"page": int, "text": str}]
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split by page delimiter (--- on its own line)
    pages = [p.strip() for p in content.split('---') if p.strip()]
    return [{"page": i+1, "text": page} for i, page in enumerate(pages)]


def load_company_faq(path: str) -> str:
    """
    Load the entire company-faq.md as a single string.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read() 