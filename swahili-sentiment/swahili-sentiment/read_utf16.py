from pathlib import Path
p = Path('tokenizer_test.out')
text = p.read_text(encoding='utf-16', errors='replace')
print(text)
