"""One-shot sweep: wrap every user-derived interpolation in the generated
SLURM job scripts with shell_arg(). Line-based and quote-aware: the #SBATCH -J
and -o lines are rebuilt explicitly (their f-strings may be single- or
double-quoted); the naive space-guard pattern and known string tokens are
rewritten with targeted regexes. $SLURM_* runtime variables are left untouched.
Idempotent: already-transformed lines do not match again.
"""

import glob
import re

NAIVE = re.compile(r"([A-Za-z_][\w\.\[\]]*) if ' ' not in \1 else f'\"\{\1\}\"'")

TOKENS = [
    'self.atlas', 'self.orientation', 'self.brain_geometry', 'self.model',
    'self.diameter', 'self.user', 'self.priority', 'experiment',
    'self.resolution_level', 'self.channel',
]

LINE_RE = re.compile(r"^(\s*f\.write\(f)(['\"])(#SBATCH -[Jo] .+?)\2\)\s*$")


def alt(text, q):
    """Quote `text` with the f-string's opposite quote character."""
    iq = '"' if q == "'" else "'"
    return f"{iq}{text}{iq}"


def rebuild(content, q):
    iq = '"' if q == "'" else "'"
    if content.startswith('#SBATCH -J {self.user}'):
        suffix = content[len('#SBATCH -J {self.user}'):]
        if suffix == '-{self.name}':
            inner = f"self.user + {iq}-{iq} + self.name"
        else:
            inner = f"self.user + {iq}{suffix}{iq}"
        return f'#SBATCH -J {{shell_arg({inner})}}'
    if content.startswith('#SBATCH -o '):
        m = re.match(r'#SBATCH -o \{(\w[\w\.]*)\}(/.*)$', content)
        expr, path = m.group(1), m.group(2)
        return f'#SBATCH -o {{shell_arg({expr} + {iq}{path}{iq})}}'
    return None


def transform_line(line):
    stripped = line.rstrip('\n')
    nl = '\n' if line.endswith('\n') else ''
    m = LINE_RE.match(stripped)
    if m:
        head, q, content = m.group(1), m.group(2), m.group(3)
        new_content = rebuild(content, q)
        if new_content:
            return f"{head}{q}{new_content}{q}){nl}"
    if "if ' ' not in" in stripped:
        return NAIVE.sub(r'shell_arg(\1)', stripped) + nl
    for token in TOKENS:
        line = re.sub(r'\{' + re.escape(token) + r'\}',
                      '{shell_arg(' + token + ')}', line)
    line = re.sub(r'\{int\(self\.with_dbscan\)\}', '{shell_arg(int(self.with_dbscan))}', line)
    line = re.sub(r'\{str\(self\.resolution\[(\d)\]\)\}', r'{shell_arg(str(self.resolution[\1]))}', line)
    line = re.sub(r'\{str\(settings\.HOME\)\}', '{shell_arg(str(settings.HOME))}', line)
    line = re.sub(r"\{(metadata_field\['key'\]\.replace\(' ', ''\))\}", r'{shell_arg(\1)}', line)
    line = re.sub(r"\{(metadata_field\['value'\]\.replace\(' ', ''\))\}", r'{shell_arg(\1)}', line)
    line = re.sub(r'(python )\{slurm_script\}', r'\1{shell_arg(slurm_script)}', line)
    return line


def transform(src):
    return ''.join(transform_line(l) for l in src.splitlines(keepends=True))


if __name__ == '__main__':
    files = [f for f in sorted(glob.glob('operations/*/main.py') + glob.glob('plugins/*/main.py')
                               + glob.glob('reader_plugins/*/main.py'))
             if 'bin/bash' in open(f).read()]
    changed = 0
    for f in files:
        src = open(f).read()
        new = transform(src)
        if new != src:
            open(f, 'w').write(new)
            changed += 1
    print(f"transformed {changed}/{len(files)} files")
