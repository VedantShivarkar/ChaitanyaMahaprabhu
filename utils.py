import re
from typing import List, Dict, Any, Tuple

def extract_evidence_from_answer(llm_answer: str) -> List[Dict[str, str]]:
