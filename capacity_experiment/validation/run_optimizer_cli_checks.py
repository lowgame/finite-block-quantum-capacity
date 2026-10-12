"""Exercise input formats and run the optimizer through its public CLI."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np


def main():
    script=Path(__file__).resolve().parents[1]/'optimize.py'
    sys.path.insert(0,str(script.parent))
    spec=importlib.util.spec_from_file_location('optimizer_cli_under_test',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        # Different lengths expose the old hardcoded n=45 raw-file loader.
        for n in (5,45,60,120):
            psi=np.zeros((2,n+1),dtype='<c16')
            psi[0,0]=1;psi[1,n]=0.7j
            raw=root/f'n{n}.c128';psi.tofile(raw)
            assert np.array_equal(module.load_search_input(raw),psi)
            checks.append({'input':'raw','n':n,'passed':True})
        for key in ('psi','purification'):
            path=root/f'{key}.npz'
            np.savez(path,**{key:psi})
            assert np.array_equal(module.load_search_input(path),psi)
            checks.append({'input':key,'n':120,'passed':True})
        np.savez(root/'ambiguous.npz',psi=psi,purification=2*psi)
        try:
            module.load_search_input(root/'ambiguous.npz')
        except ValueError:
            checks.append({'input':'ambiguous_NPZ_rejected','passed':True})
        else:
            raise AssertionError('Conflicting coefficient arrays were accepted.')
        # Run from an isolated copy: the optimizer's cache must not touch
        # archived data or the checkout used by other tests.
        copied=root/'optimize.py';copied.write_bytes(script.read_bytes())
        (root/'core.py').write_bytes(script.with_name('core.py').read_bytes())
        for kind in ('complex','real'):
            output=root/kind
            command=[sys.executable,str(copied),'--input',str(root/'n5.c128'),
                     '--p','0.19','--n','6','--maxiter','2','--output',str(output)]
            if kind=='real':
                real=np.zeros((2,6),dtype='<c16');real[0,0]=1;real[1,5]=0.7
                real.tofile(root/'n5.c128')
                command.append('--real')
            completed=subprocess.run(command,capture_output=True,text=True)
            if completed.returncode:
                raise AssertionError(completed.stderr)
            report=json.loads(output.with_suffix('.json').read_text())
            with np.load(output.with_suffix('.npz'),allow_pickle=False) as archive:
                state=archive['psi']
            assert state.shape==(2,7) and np.isfinite(state).all()
            assert abs(float(np.vdot(state,state).real)-1)<1e-12
            assert np.isfinite(report['ic_bits']) and report['evaluations']>0
            assert abs(report['trace']-1)<1e-10
            assert report['ic_bits']>=report['initial_ic']-1e-12
            checks.append({'input':'CLI_'+kind+'_lifted','passed':True,
                           'n':report['n'],'evaluations':report['evaluations'],
                           'ic_bits':report['ic_bits']})
    print(json.dumps({'passed':True,'checks':checks},indent=2))


if __name__=='__main__':
    main()
