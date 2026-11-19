"""Compare four symmetric absorption models on identical held frequency blocks."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.special import voigt_profile

from compare_asymmetry import harmonic
from survey import OUT, fit, inputs


def lorentz_harmonic(offset, gamma, excursion):
    z = offset + 1j*gamma
    # Exact first harmonic of gamma/(u**2 + gamma**2), including the
    # derivative limit. The product picks the square root continuous above
    # the real axis. The rationalized form stays stable at small excursion.
    root = np.sqrt(z-excursion)*np.sqrt(z+excursion)
    return (-2j/(root*(z+root))).real


def fit_lorentz(x,y,train,finite,width):
    def solve(q):
        shape = lorentz_harmonic(x-q[0],q[1],q[2] if finite else 0.)
        m = np.column_stack((shape,np.ones_like(x),x))
        beta = np.linalg.lstsq(m[train],y[train],rcond=None)[0]
        return m@beta-y, beta

    extent = max(4.,1.5*width)
    limit = max(10.,4*width)
    lower = [max(x.min(),-extent),.02] + ([0.] if finite else [])
    upper = [min(x.max(),extent),limit] + ([limit] if finite else [])
    starts = [[0,width]]
    if finite:
        starts = [[0,width,.1*width],[0,.55*width,.9*width],[0,.18*width,1.35*width]]
    fits = [least_squares(lambda q:solve(q)[0][train],q,bounds=(lower,upper),
                          max_nfev=700,ftol=2e-9,xtol=2e-9) for q in starts]
    best = min((r for r in fits if r.success),key=lambda r:r.cost)
    residual,beta = solve(best.x)
    return [best.x[0],best.x[1],0,best.x[2] if finite else 0],residual,beta


def fit_model(x,y,train,voigt,finite,width):
    if not voigt:
        return fit_lorentz(x,y,train,finite,width)
    seeds = [[0,.4*width,.55*width],[0,.95*width,.12*width]]
    if finite:
        seeds = [seeds[0]+[.85*width],seeds[1]+[.3*width],
                 [0,.18*width,.18*width,1.35*width]]
    q,residual,beta = fit(x,y,train,False,finite,width,seeds)
    return list(q) if finite else [*q,0.],residual,beta


def analyze(item):
    task,path,ref,width,px,pz,volts,hz,warning = item
    f,y = np.loadtxt(path).T
    x = (f-ref)/1000
    train_all = np.ones(len(y),bool)
    blocks = [(x>=lo) & ((x<=6) if lo==4 else (x<lo+2)) for lo in range(-6,6,2)]
    blocks = [held for held in blocks if held.any()]
    rows = []
    for model,voigt,finite in [("Lorentz",False,False),("Voigt",True,False),
                               ("finite_Lorentz",False,True),("finite_Voigt",True,True)]:
        q,residual,beta = fit_model(x,y,train_all,voigt,finite,width)
        center,gamma,sigma,excursion = q
        half = voigt_profile(0,sigma,gamma)/2
        fwhm = 2*brentq(lambda u:voigt_profile(u,sigma,gamma)-half,0,10*(sigma+gamma))
        errors = []
        for held in blocks:
            _,e,_ = fit_model(x,y,~held,voigt,finite,width)
            errors.extend(e[held])
        rows.append(dict(task=task,filename=path.name,x_cm=px,z_cm=pz,voltage_V=volts,
                         modulation_Hz=hz,model=model,baseline_warning=warning,
                         f0_Hz=ref+center*1000,gamma_kHz=gamma,sigma_kHz=sigma,
                         excursion_kHz=excursion,absorption_fwhm_kHz=fwhm,
                         amplitude=beta[0],background=beta[1],slope_per_kHz=beta[2],
                         reference_Hz=ref,n=len(y),n_held=len(errors),n_blocks=len(blocks),
                         full_rms=np.sqrt(np.mean(residual**2)),held_rms=np.sqrt(np.mean(np.square(errors)))))
    return rows


def plot_examples(table, selected):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":["Yu Gothic","DejaVu Sans"]})
    paths = {item[1].name:item[1] for item in selected}
    examples = ["20261110_x3.0_z3.5_01.dat","20261104_v4.5_f280_01.dat"]
    titles = ["課題2　11月10日 (3.0, 3.5) cm","課題3　4.5 V・280 Hz"]
    fig,axes = plt.subplots(2,2,figsize=(10.5,6.2),height_ratios=[2,1],sharex="col",
                            layout="constrained")
    for col,(filename,title) in enumerate(zip(examples,titles)):
        top,bottom = axes[:,col]
        f,y = np.loadtxt(paths[filename]).T
        rows = table[table.filename==filename].set_index("model")
        ref = rows.loc["Lorentz","reference_Hz"]
        x = (f-ref)/1000
        show = abs(x)<=8
        top.plot(x[show],y[show],".",c=".4",label="測定値")
        grid = np.linspace(max(-8,x.min()),min(8,x.max()),501)
        models = [("Lorentz","微分ローレンツ","#786856","--"),
                  ("Voigt","微分Voigt","#00689f","-")]
        if col==1:
            models.append(("finite_Voigt","有限変調Voigt","#c14323","-"))
        for name,label,color,style in models:
            r = rows.loc[name]
            def predict(t):
                u = t-(r.f0_Hz-ref)/1000
                h = (harmonic(u,r.gamma_kHz,r.sigma_kHz,r.excursion_kHz,nodes=128).real
                     if r.sigma_kHz>0 else lorentz_harmonic(u,r.gamma_kHz,r.excursion_kHz))
                return r.amplitude*h+r.background+r.slope_per_kHz*t
            np.testing.assert_allclose(np.sqrt(np.mean((y-predict(x))**2)),r.full_rms,rtol=1e-5)
            top.plot(grid,predict(grid),style,color=color,lw=1.7,label=label)
            bottom.plot(x[show],y[show]-predict(x[show]),style,color=color,lw=1,marker=".",ms=3)
        top.set_title(title)
        top.legend(fontsize=9)
        bottom.axhline(0,lw=.6,color=".5")
        bottom.set_xlabel("ローレンツ中心からの周波数差 (kHz)")
        top.grid(alpha=.15)
        bottom.grid(alpha=.15)
    axes[0,0].set_ylabel("ロックイン出力（記録単位）")
    axes[1,0].set_ylabel("測定値 − フィット")
    fig.savefig(OUT/"symmetric_examples.png",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    import sys
    selected = [item for item in inputs() if item[0] in (2,3)]
    if "--plot-only" in sys.argv:
        plot_examples(pd.read_csv(OUT/"symmetric_models.csv"),selected)
        sys.exit()
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = [row for group in pool.map(analyze,selected) for row in group]
    table = pd.DataFrame(rows)
    assert len(table)==420 and table.held_rms.notna().all()
    table.to_csv(OUT/"symmetric_models.csv",index=False,encoding="utf-8-sig")
    plot_examples(table,selected)
