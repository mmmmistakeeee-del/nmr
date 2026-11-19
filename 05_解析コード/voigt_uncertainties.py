"""Conditional block-wild residual bootstrap on the accepted Voigt models."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import zlib
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import voigt_profile
from survey import OUT, inputs, fit
from compare_asymmetry import harmonic

DEST = OUT.parent/"図表"
CACHE = OUT.parent/"再フィット"
NBOOT = 120
METRICS = ["f0_MHz","field_mT","fwhm_kHz","pp_kHz","height","gamma_kHz","sigma_kHz","excursion_kHz"]


def quantities(q,amp,ref):
    c,g,s = q[:3]
    a = q[3] if len(q)==4 else 0.
    half = voigt_profile(0,s,g)/2
    width = 2*brentq(lambda u:voigt_profile(u,s,g)-half,0,10*(g+s))
    def line(u):
        return harmonic(np.atleast_1d(u),g,s,a,nodes=128).real
    peak = minimize_scalar(lambda u:line(u)[0],bounds=(0,2*a+3*width),
                           method="bounded",options={"xatol":2e-7}).x
    f0 = (ref+1000*c)/1e6
    return [f0,1000*f0/42.5759,width,2*peak,2*abs(amp*line(peak)[0]),g,s,a]


def run(args):
    item,r = args
    task,path,ref,width,px,pz,v,hz,warn = item
    f,y = np.loadtxt(path).T
    x = (f-ref)/1000
    finite = task==3
    q = [(r.f0_Hz-ref)/1000,r.gamma_kHz,r.sigma_kHz]
    if finite:
        q.append(r.excursion_kHz)
    shape = harmonic(x-q[0],q[1],q[2],r.excursion_kHz,nodes=128).real
    mu = r.background+r.slope_per_kHz*x+r.amplitude*shape
    residual = y-mu
    np.testing.assert_allclose(np.sqrt(np.mean(residual**2)),r.full_rms,rtol=3e-5,atol=1e-7)
    obs = quantities(q,r.amplitude,ref)
    n = len(y)
    block = int(np.ceil(n**(1/3)))
    rng = np.random.default_rng(zlib.crc32(path.name.encode())+20261119)
    err = (residual-np.mean(residual))*np.sqrt(n/(n-len(q)-3))
    train = np.ones(n,dtype=bool)
    samples = []
    for _ in range(NBOOT):
        groups = (np.arange(n)+rng.integers(0,block))//block
        signs = rng.choice([-1,1],size=groups[-1]+1)
        seeds = [q]
        if finite:
            # Competing small- and large-excursion starts expose width tradeoffs.
            alt = [q[0],.45*width,.6*width,.1*width if q[3]>.3*width else width]
            seeds.append(alt)
        try:
            p,_,b = fit(x,mu+err*signs[groups],train,False,finite,width,seeds)
            samples.append(quantities(p,b[0],ref))
        except (ValueError,RuntimeError,np.linalg.LinAlgError):
            continue
    if len(samples)<.9*NBOOT:
        raise RuntimeError(f"{path.name}: only {len(samples)} valid fits")
    values = np.array(samples)
    row = dict(task=task,file=path.name,date=path.name[:8],x_cm=px,z_cm=pz,voltage_V=v,
               modulation_Hz=hz,model=r.model,n=n,block=block,
               refits=len(samples),rms=r.full_rms,cv_rms=r.held_rms)
    for j,key in enumerate(METRICS):
        row[key] = obs[j]
        row[key+"_sd"] = values[:,j].std(ddof=1)
        row[key+"_p16"],row[key+"_p84"] = np.quantile(values[:,j],[.16,.84])
    pd.DataFrame([row]).to_csv(CACHE/(path.stem+".csv"),index=False,encoding="utf-8-sig")
    return path.name


if __name__=="__main__":
    CACHE.mkdir(parents=True,exist_ok=True)
    DEST.mkdir(exist_ok=True)
    fits = pd.read_csv(OUT/"symmetric_models.csv")
    fits = fits[((fits.task==2)&(fits.model=="Voigt"))|((fits.task==3)&(fits.model=="finite_Voigt"))]
    by_name = fits.set_index("filename")
    jobs = [(i,by_name.loc[i[1].name]) for i in inputs() if i[0]!=1]
    assert len(jobs)==105
    order = sorted(jobs,key=lambda j:-j[0][0])
    with ProcessPoolExecutor(max_workers=8) as pool:
        running = [pool.submit(run,j) for j in order if not (CACHE/(j[0][1].stem+".csv")).exists()]
        for future in as_completed(running):
            future.result()
    result = pd.concat([pd.read_csv(CACHE/(j[0][1].stem+".csv")) for j in jobs],ignore_index=True)
    assert result.file.is_unique and len(result)==105 and result.refits.min()>=.9*NBOOT
    result.to_csv(DEST/"Voigt推定値と不確かさ.csv",index=False,encoding="utf-8-sig")
