"""Copy HRSID image bytes into the exact archived split layout; never resplit."""
import argparse,hashlib,json,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-images',type=Path,required=True,help='Flat or nested directory containing original HRSID tiles')
    p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[2]/'datasets/hrsid')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    archive=Path(__file__).resolve().parents[2]/'datasets/hrsid'
    rows=json.loads((archive/'image_identity_manifest.json').read_text(encoding='utf-8'))
    wanted={r['original_file_name'] for r in rows};index={}
    for f in args.source_images.rglob('*'):
        if f.is_file() and f.name in wanted:index.setdefault(f.name,[]).append(f)
    selected=[]
    for r in rows:
        matching=[f for f in index.get(r['original_file_name'],[]) if sha(f)==r['original_sha256']]
        if not matching:raise ValueError(f"Missing original image or SHA256 mismatch: {r['original_file_name']}")
        selected.append((r,matching[0]))
    if not args.check_only:
        for r,src in selected:
            dst=args.output_root/r['image_relative_path'];dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():
                if sha(dst)!=r['training_sha256']:raise FileExistsError(f'Conflicting destination: {dst}')
            else:shutil.copy2(src,dst)
    print(json.dumps({'verified':len(selected),'copied':0 if args.check_only else len(selected),'resplit':False}))

if __name__=='__main__':main()

