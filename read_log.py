from pathlib import Path

for name in ['train_afriberta4.log', 'train_afriberta3.log', 'train_afriberta2.log', 'train_afriberta.log']:
    p = Path(name)
    if not p.exists():
        continue
    text = p.read_text(encoding='utf-16', errors='replace')
    lines = text.splitlines()
    print('FILE', name)
    print('EXISTS', p.exists(), 'SIZE', p.stat().st_size, 'LINES', len(lines))
    print('\n'.join(lines[-40:]))
    print('\n' + '='*80 + '\n')
