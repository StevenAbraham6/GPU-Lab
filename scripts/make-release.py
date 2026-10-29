"""Build a source tarball without local runs, caches or credentials."""
import hashlib
import tarfile
from pathlib import Path

root=Path(__file__).resolve().parent.parent
output=root/'dist';output.mkdir(exist_ok=True)
target=output/'gpu-lab-0.1.0.tar.gz'
include=['gpulab','scripts','examples','tests','pyproject.toml','README.md','LICENSE','install.sh','docs','.github','.gitignore','.gitattributes']
with tarfile.open(target,'w:gz') as archive:
    for name in include:
        path=root/name
        if not path.exists():continue
        files=sorted(path.rglob('*')) if path.is_dir() else [path]
        for file in files:
            if not file.is_file() or '__pycache__' in file.parts or file.suffix=='.pyc':continue
            archive.add(file,arcname='gpu-lab-0.1.0/'+str(file.relative_to(root)))
digest=hashlib.sha256(target.read_bytes()).hexdigest()
target.with_suffix(target.suffix+'.sha256').write_text(digest+'  '+target.name+'\n')
print(target)
