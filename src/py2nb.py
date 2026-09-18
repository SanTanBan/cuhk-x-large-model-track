"""Convert a `# %%` cell-delimited Python script into a Jupyter notebook.

Cells get stable `id`s: nbformat warns that missing ids will become a hard error, and
Kaggle's runner already emits that warning.
"""
import json, re, sys, io, hashlib


def _cell_id(i, src):
    return hashlib.sha1(('%d:%s' % (i, src)).encode('utf-8')).hexdigest()[:8]

def convert(src_path, out_path):
    src = io.open(src_path, encoding='utf-8').read()
    cells = []
    for chunk in src.split('# %%'):
        c = chunk.strip('\n')
        if not c.strip():
            continue
        if c.lstrip().startswith('[markdown]'):
            body = re.sub(r'^\s*\[markdown\]\n', '', c)
            lines = []
            for l in body.split('\n'):
                lines.append(l[2:] if l.startswith('# ') else (l[1:] if l.startswith('#') else l))
            cells.append({'cell_type': 'markdown', 'metadata': {},
                          'source': '\n'.join(lines).splitlines(True)})
        else:
            cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                          'outputs': [], 'source': c.splitlines(True)})
    for i, c in enumerate(cells):
        c['id'] = _cell_id(i, ''.join(c['source']))
    nb = {'cells': cells,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                      'name': 'python3'},
                       'language_info': {'name': 'python', 'version': '3.11'}},
          'nbformat': 4, 'nbformat_minor': 5}
    io.open(out_path, 'w', encoding='utf-8').write(json.dumps(nb, indent=1))
    return len(cells)

if __name__ == '__main__':
    n = convert(sys.argv[1], sys.argv[2])
    print('wrote %s with %d cells' % (sys.argv[2], n))
