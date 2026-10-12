from pathlib import Path
import subprocess, sys, time, json, platform, resource, hashlib
from datetime import datetime, timezone
base=Path('/workspace/scratch/6a8085e11ed6/capacity_experiment/baseline')
art=base/'zenodo_v1_0/artifact'
command=[sys.executable,'verifier/verify.py','certificates/A-p16239-250000.json','1df2e00c06b2614cd04a5bfc7bede040da2d3fc1dd49e7138e201c101aab266f','16239/250000']
meta={'command':command,'cwd':str(art),'started_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'log':str(base/'verify_A_positive.log'),'noise_convention':'per-Pauli r=16239/250000; total p=3r=0.194868','verifier_sha256':hashlib.sha256((art/'verifier/verify.py').read_bytes()).hexdigest(),'certificate_sha256':hashlib.sha256((art/'certificates/A-p16239-250000.json').read_bytes()).hexdigest()}
t0=time.perf_counter()
with (base/'verify_A_positive.log').open('wb') as log:
 process=subprocess.Popen(command,cwd=art,stdout=log,stderr=subprocess.STDOUT)
 meta['pid']=process.pid
 (base/'verify_A_positive_run.json').write_text(json.dumps(meta,indent=2))
 print(json.dumps(meta),flush=True)
 rc=process.wait()
meta.update(returncode=rc,finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-t0,cpu_user_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,cpu_system_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
meta['output']=(base/'verify_A_positive.log').read_text()
(base/'verify_A_positive_run.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta),flush=True)
