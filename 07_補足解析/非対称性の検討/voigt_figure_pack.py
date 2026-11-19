"""Build the standalone Japanese figure and table pack from saved fits."""
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib import ticker
import numpy as np
import pandas as pd
from compare_symmetric import lorentz_harmonic
from compare_asymmetry import NMR, OUT, harmonic, TASK1, RESULTS, TASK3
from survey import T2, position_fit_display
from voigt_uncertainties import DEST

plt.rcParams.update({"font.family":["Yu Gothic","DejaVu Sans"],"font.size":10,
                     "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
BLUE="#145784"
ORANGE="#b34a20"
INK="#153348"
GRAY="#586671"
FLAGS={"correlated_residuals":"残差相関","weak_signal":"弱信号","short_baseline_range":"背景範囲不足"}
DATE_MARK={20261021:"o",20261027:"s",20261028:"^",20261104:"D",20261110:"P"}
STEM="20261020_KITtest01"
FIGS=[("図1_課題1_波形と残差.png","課題1　共鳴波形と残差"),
      ("図2_課題2_位置分布.png","課題2　磁場・線幅・信号強度の位置分布"),
      ("図3_課題2_不確かさ.png","課題2　位置ごとの推定の不確かさ"),
      ("図4_課題3_電圧依存.png","課題3　変調電圧を変えたときの応答"),
      ("図5_課題3_周波数依存.png","課題3　変調周波数を変えたときの応答"),
      ("図6_課題3_代表波形.png","課題3　4.5 V・280 Hzの波形と残差")]
CAPTIONS=[
    "10月20日の上り掃引。実測点に微分ローレンツ型と微分Voigt型を当てはめた。下段は各周波数での測定値とフィットの差。全波形をフィットし、共鳴付近を拡大した。",
    "定量採用78座標の微分Voigtフィット。色は各物理量、形は測定日を示す。磁場は共鳴周波数から換算した。白抜きは磁場のみの参考点、灰色の×は推定保留点を示す。",
    "図2と同じ定量採用78座標。色は120回のブロック単位の残差再標本化による標準不確かさ。周波数校正や測定日間の系統差は含まない。",
    "変調周波数280 Hz、電圧10条件。有限変調Voigt型で、変調前の吸収線の半値全幅と観測基本波の山谷間隔を別々に求めた。誤差棒は標準不確かさ。電圧は装置表示値。",
    "電圧表示1.3 V。20、60 Hzは各パネル下端に参考条件の位置だけを示した。残り18条件の誤差棒は有限変調Voigt型の標準不確かさ。120、180、240 Hzは背景に構造がある。",
    "微分ローレンツ型、微分Voigt型、有限変調Voigt型を同じ全測定点に当てはめた。共鳴付近を拡大し、下段に測定値と各モデルの差を示す。"]


def save(fig,index):
    fig.savefig(DEST/FIGS[index][0],dpi=210,bbox_inches="tight",pad_inches=.14,facecolor="white")
    plt.close(fig)


def maps(d,errors=False):
    metrics = ([("f0_MHz_sd",1e6,"共鳴周波数の標準不確かさ (Hz)"),
                ("fwhm_kHz_sd",1,"半値全幅の標準不確かさ (kHz)"),
                ("height_sd",1,"山谷差の標準不確かさ（記録単位）")] if errors else
               [("field_mT",1,"磁束密度 (mT)"),("fwhm_kHz",1,"吸収線の半値全幅 (kHz)"),
                ("height",1,"信号の山谷差（記録単位）")])
    fig,axes=plt.subplots(1,3,figsize=(11.8,4.1))
    for col,(ax,(key,scale,title)) in enumerate(zip(axes,metrics)):
        vals=d[key]*scale
        norm=plt.Normalize(vals.min(),vals.max())
        cmap="viridis"
        for date,sub in d.groupby("date"):
            ax.scatter(sub.x_cm,sub.z_cm,c=sub[key]*scale,norm=norm,cmap=cmap,s=50,
                       marker=DATE_MARK[date],edgecolors="#454545",linewidths=.45)
        if not errors:
            ax.scatter(2.5,6.5,marker="x",s=58,c=GRAY)
            if col==0:
                ax.scatter(2.5,4.5,marker="o",s=48,facecolors="none",edgecolors=GRAY)
        ax.set(aspect="equal",xlim=(2.1,6.9),ylim=(2.1,6.9),xticks=[2.5,3.5,4.5,5.5,6.5],
               yticks=[2.5,3.5,4.5,5.5,6.5],xlabel="x (cm)",ylabel="z (cm)",title=title)
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,orientation="horizontal",
                        fraction=.07,pad=.17)
        cb.formatter=ticker.ScalarFormatter(useOffset=False)
        cb.locator=ticker.MaxNLocator(4)
        cb.update_ticks()
    handles=[Line2D([],[],color=GRAY,ls="",marker=m,label=f"{str(date)[4:6]}/{str(date)[6:]}")
             for date,m in DATE_MARK.items()]
    if not errors:
        handles.extend([Line2D([],[],ls="",color=GRAY,marker="o",mfc="none",label="磁場のみ参考"),
                        Line2D([],[],ls="",color=GRAY,marker="x",label="推定保留")])
    fig.legend(handles=handles,ncol=len(handles),loc="upper center",frameon=False)
    fig.subplots_adjust(top=.82,bottom=.13,wspace=.34)
    save(fig,2 if errors else 1)


def trends(d,frequency):
    xcol="modulation_Hz" if frequency else "voltage_V"
    sub=d[d.voltage_V.eq(1.3) if frequency else d.modulation_Hz.eq(280)].sort_values(xcol)
    fig,axes=plt.subplots(2,2,figsize=(9.5,6.4))
    for ax,(key,factor,label) in zip(axes.flat,[("f0_MHz",1000,"共鳴周波数 − 16 MHz (kHz)"),
            ("fwhm_kHz",1,"変調前の吸収線の半値全幅 (kHz)"),("pp_kHz",1,"観測波形の山谷間隔 (kHz)"),
            ("height",1,"信号の山谷差（記録単位）")]):
        for marked,g in sub.groupby(sub.baseline_flag.fillna("").ne("")):
            val=g[key]-(16 if key=="f0_MHz" else 0)
            ax.errorbar(g[xcol],factor*val,yerr=factor*g[key+"_sd"],fmt="^" if marked else "o",
                        color=ORANGE if marked else BLUE,ms=5.5,capsize=2,lw=.9)
        if frequency:
            ax.scatter([20,60],[.025,.025],transform=ax.get_xaxis_transform(),c=GRAY,marker="x",s=32)
            ax.set_xlim(0,420)
        ax.set(xlabel="変調周波数 (Hz)" if frequency else "変調電圧 (V)",ylabel=label)
        ax.grid(axis="y",alpha=.23)
    handles=[Line2D([],[],ls="",marker="o",c=BLUE,label="通常"),
             Line2D([],[],ls="",marker="^",c=ORANGE,label="元の解析で注意あり")]
    if frequency:
        handles.append(Line2D([],[],ls="",marker="x",c=GRAY,label="参考条件（縦位置に数値の意味なし）"))
    fig.legend(handles=handles,loc="upper center",ncol=len(handles),frameon=False)
    fig.tight_layout(rect=(0,0,1,.94),h_pad=1.5,w_pad=1.5)
    save(fig,4 if frequency else 3)


def example(models):
    name="20261104_v4.5_f280_01.dat"
    f,y=np.loadtxt(TASK3/"20261104"/name).T
    fig,(top,bottom)=plt.subplots(2,1,figsize=(8.2,5.1),sharex=True,height_ratios=[2,1],
                                 layout="constrained")
    group=models[models.filename==name].set_index("model")
    ref=group.loc["Lorentz","reference_Hz"]
    x=(f-ref)/1000
    use=abs(x)<=9
    grid=np.linspace(-9,9,600)
    top.plot(x[use],y[use],".",color=".35",label="測定値")
    for key,label,c,style in [("Lorentz","微分ローレンツ",GRAY,":"),("Voigt","微分Voigt",BLUE,"--"),
                              ("finite_Voigt","有限変調Voigt",ORANGE,"-")]:
        r=group.loc[key]
        def pred(t):
            u=t-(r.f0_Hz-ref)/1000
            h=(harmonic(u,r.gamma_kHz,r.sigma_kHz,r.excursion_kHz,128).real if r.sigma_kHz else
               lorentz_harmonic(u,r.gamma_kHz,r.excursion_kHz))
            return r.amplitude*h+r.background+r.slope_per_kHz*t
        np.testing.assert_allclose(np.sqrt(np.mean((y-pred(x))**2)),r.full_rms,rtol=1e-5)
        top.plot(grid,pred(grid),style,c=c,label=label,lw=1.6)
        bottom.plot(x[use],(y-pred(x))[use],style,marker=".",c=c,lw=1)
    top.set_ylabel("ロックイン出力（記録単位）")
    bottom.set(xlabel=f"{ref/1e6:.6f} MHz からの周波数差 (kHz)",ylabel="測定値 − フィット")
    top.legend(frameon=False,fontsize=9)
    for ax in (top,bottom):
        ax.axhline(0,color=".7",lw=.7)
        ax.grid(alpha=.15)
    save(fig,5)


def page(pdf,title,subtitle=""):
    fig=plt.figure(figsize=(11.69,8.27),facecolor="white")
    fig.text(.065,.935,title,fontsize=21,weight="bold",color=INK,va="top")
    if subtitle:
        fig.text(.066,.864,subtitle,fontsize=10.5,color=GRAY,va="top")
    fig.add_artist(Line2D([.065,.936],[.825,.825],color="#cbd8df",lw=1))
    fig.text(.937,.032,str(pdf.get_pagecount()+1),ha="right",color=GRAY,fontsize=9)
    return fig


def lines(fig,y,items,size=11,step=.041):
    for item in items:
        fig.text(.07,y,item,va="top",color=INK,fontsize=size)
        y-=step


def table_at(fig,columns,rows,box,fontsize=9,widths=None):
    ax=fig.add_axes(box)
    ax.axis("off")
    t=ax.table(cellText=rows,colLabels=columns,cellLoc="center",colLoc="center",bbox=(0,0,1,1),
               colWidths=widths)
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(.5)
        cell.set_facecolor("#dbe8ee" if i==0 else ("#f1f5f7" if i%2 else "#ffffff"))
        cell.set_text_props(color=INK,weight="bold" if i==0 else "normal")


def savepage(pdf,fig):
    pdf.savefig(fig)
    plt.close(fig)


def flag_text(row):
    flags=[FLAGS.get(k,k) for k in str(row.baseline_flag).split(";") if k and k!="nan"]
    if row.task==3 and row.modulation_Hz in (120,180,240):
        flags.append("背景構造")
    if row.fwhm_kHz_sd>=.25*row.fwhm_kHz:
        flags.append("幅の不確かさ大")
    if row.gamma_kHz<=.0201 or row.sigma_kHz<=.0401 or (row.task==3 and row.excursion_kHz<=.0011):
        flags.append("形状量が下限")
    return "・".join(flags)


def compact_flag(row):
    return " ".join(tag for word,tag in [("弱信号","弱"),("残差相関","相"),("背景範囲不足","短"),
                                         ("背景構造","背"),("幅の不確かさ大","幅"),("形状量が下限","限")]
                    if word in row.notes) or "—"


def paired(mean,sd,digits):
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def exclusions():
    pos=pd.read_csv(T2/"解析II/results_with_additions/fit_results.csv")
    rows=[]
    for _,r in pos.iterrows():
        status=position_fit_display(r)
        if status=="estimate":
            continue
        if r.x==7 or r.z in (7,2):
            reason="設定した作図範囲外"
        elif status=="reference":
            reason="背景モデルで線幅が大きく変わる。磁場図に白抜きで位置だけを示す。"
        elif status=="withheld":
            reason="弱信号。範囲変更で中心が8.53 kHz移動し、位置のみ×で示す。"
        else:
            reason="指定の除外掃引。共鳴が不明瞭。"
        rows.append([2,r.filename,r.x,r.z,"",status,reason])
    mod=pd.read_csv(RESULTS/"fit_results.csv")
    for _,r in mod[mod.modulation_Hz.isin([20,60])].iterrows():
        why="弱信号・背景範囲不足" if r.modulation_Hz==20 else "周期的背景・中心の範囲依存"
        rows.append([3,r.filename,"","",r.modulation_Hz,"reference",why])
    return pd.DataFrame(rows,columns=["課題","測定ファイル","x (cm)","z (cm)","変調周波数 (Hz)","扱い","理由"])


def build():
    dest=DEST
    d=pd.read_csv(dest/"Voigt推定値と不確かさ.csv")
    d["notes"]=[flag_text(r) for r in d.itertuples()]
    pos=d[d.task==2].sort_values(["date","z_cm","x_cm"])
    mod=d[d.task==3].sort_values(["voltage_V","modulation_Hz"])
    assert len(pos)==78 and len(mod)==27 and d.file.is_unique
    models=pd.read_csv(OUT/"symmetric_models.csv")
    t1dir=TASK1/"解析I"
    task1=pd.read_csv(t1dir/(STEM+"_uncertainty_parameters.csv")).set_index("parameter")
    shutil.copyfile(NMR/"レポート/figures/task1_fit_residuals.png",dest/FIGS[0][0])
    shutil.copyfile(t1dir/(STEM+"_uncertainty_parameters_ja.csv"),dest/"課題1_推定値と不確かさ.csv")
    shutil.copyfile(OUT/"symmetric_models.csv",dest/"4モデルの比較_全420件.csv")
    excl=exclusions()
    assert len(excl)==7
    excl.to_csv(dest/"参考点と除外点.csv",index=False,encoding="utf-8-sig")
    maps(pos)
    maps(pos,True)
    trends(mod,False)
    trends(mod,True)
    example(models)
    for task,g in [(2,pos),(3,mod)]:
        j=pd.DataFrame({"測定日":g.date,"ファイル名":g.file})
        for key,label,sdcol in [("x_cm","x (cm)",False),("z_cm","z (cm)",False),
                ("voltage_V","変調電圧 (V)",False),("modulation_Hz","変調周波数 (Hz)",False),
                ("f0_MHz","共鳴周波数 (MHz)",True),("field_mT","磁束密度 (mT)",True),
                ("fwhm_kHz","変調前の吸収線半値全幅 (kHz)",True),("pp_kHz","観測波形の山谷間隔 (kHz)",True),
                ("height","信号の山谷差（記録単位）",True),("gamma_kHz","ローレンツ半値半幅 (kHz)",True),
                ("sigma_kHz","ガウス標準偏差 (kHz)",True),("excursion_kHz","変調片振幅 (kHz)",True)]:
            if g[key].isna().all():
                continue
            j[label]=g[key]
            if sdcol:
                j["標準不確かさ："+label]=g[key+"_sd"]
        j["再フィット回数"]=g.refits
        j["注意点"]=g.notes
        j.to_csv(dest/f"課題{task}_主解析表.csv",index=False,encoding="utf-8-sig")
    risk=d[d.notes.ne("")][["task","file","notes","fwhm_kHz","fwhm_kHz_sd"]].copy()
    risk["線幅の相対標準不確かさ (%)"]=100*risk.fwhm_kHz_sd/risk.fwhm_kHz
    risk.rename(columns={"task":"課題","file":"測定ファイル","notes":"注意点","fwhm_kHz":"線幅 (kHz)",
                         "fwhm_kHz_sd":"線幅の標準不確かさ (kHz)"}).to_csv(dest/"注意を要する波形.csv",index=False,encoding="utf-8-sig")
    with PdfPages(dest/"Voigt解析_図表集.pdf",metadata={"Title":"核磁気共鳴実験　Voigt解析の図表",
        "Author":"物理学実験","Subject":"課題1～3の図、全採用条件の数値表、不確かさ"}) as pdf:
        fig=page(pdf,"核磁気共鳴実験　Voigt解析の図表","課題1・2　微分Voigt型　／　課題3　有限変調Voigt型")
        lines(fig,.77,["課題1　共鳴中心、線幅、観測波形の山谷間隔"],size=14)
        t1rows=[]
        for key,label in [("f0_MHz","共鳴周波数"),("absorption_Voigt_FWHM_kHz","吸収線の半値全幅"),
                          ("observed_peak_to_peak_kHz","観測波形の山谷間隔"),("gamma_HWHM_kHz","ローレンツ半値半幅"),
                          ("static_sigma_kHz","ガウス標準偏差")]:
            r=task1.loc[key]
            t1rows.append([label,f"{r.estimate:.6f}" if key=="f0_MHz" else f"{r.estimate:.3f}",
                           f"{r.bootstrap_SD:.6f}" if key=="f0_MHz" else f"{r.bootstrap_SD:.3f}",r.unit])
        table_at(fig,["物理量","推定値","標準不確かさ","単位"],t1rows,[.07,.45,.84,.245],11)
        lines(fig,.39,["課題2　78座標中64座標で、微分Voigt型の区間予測誤差が微分ローレンツ型より小さい。",
                       "　　　  誤差の減少率の中央値は19.9%。共鳴中心の差の絶対値の中央値は27 Hz。",
                       "課題3　定量比較の27条件すべてで微分Voigt型が微分ローレンツ型より予測誤差を低減。",
                       "　　　  4.5 V・280 Hzでは有限変調の効果が大きく、観測波形の山谷間隔は5.07 kHz。",
                       "2～7ページに図1～6、8ページに解析条件、9ページ以降に全採用条件の数値を示す。"],
                       size=11,step=.054)
        savepage(pdf,fig)
        for idx,(filename,title) in enumerate(FIGS):
            fig=page(pdf,f"図{idx+1}　{title}")
            ax=fig.add_axes([.055,.19,.89,.62])
            ax.imshow(plt.imread(dest/filename))
            ax.axis("off")
            import textwrap
            caption="\n".join(textwrap.wrap(CAPTIONS[idx],width=53))
            fig.text(.07,.155,caption,ha="left",va="top",color=INK,fontsize=10.5,linespacing=1.55)
            savepage(pdf,fig)
        fig=page(pdf,"解析条件と結果の読み方","中心周波数・半値全幅・山谷間隔には、それぞれ異なる定義を用いた。")
        lines(fig,.78,[
          "波形　全条件に直線背景を加えた。背景を差し引いたフィット曲線の両極値から、",
          "　　　周波数差を山谷間隔、出力差を信号強度とした。吸収線幅は変調前のVoigt吸収線の半値全幅。",
          "磁場　水素核の換算係数42.5759 MHz/Tを使用した。位置図は複数日の結果で、日付を跨ぐ系統差がある。",
          "　　　別途比較した同一4座標では、11月10日の中心が10月27日より平均9.675 kHz高かった。",
          "誤差　課題1は既存の雑音モデルによる300回の再フィット。最良解の変調幅はゼロで、",
          "　　　既存の不確かさ評価では再フィット時に変調幅も自由にした。課題2・3は各120回。",
          "　　　各掃引の中心化した残差に、連続する測定点のブロックごとに無作為な符号を掛け、",
          "　　　フィット曲線に戻して再推定した。ブロック長は点数の立方根を切り上げ、開始位置を毎回ずらした。",
          "　　　誤差棒と表の ± は再推定値の標準偏差。モデル選択、周波数・変調振幅の校正誤差は含まない。",
          "検証　同じ採用データと背景で4モデルを比較。基準中心の前後6 kHzを2 kHzずつ順に外し、",
          "　　　残った点でフィットして、外した全点の予測誤差の二乗平均平方根を求めた。",
          "解釈　有限変調幅、ガウス幅、ローレンツ幅は相互に補償し得る。特に境界解では個別の幅の解釈に注意。",
          "　　　フィットしたローレンツ幅を試料固有の緩和時間へ直接換算するには追加の根拠が要る。",
          "保留　課題2の (2.5, 6.5) cm は弱信号、(2.5, 4.5) cm は背景依存。課題3の20・60 Hzは定量比較外。",
          "　　　課題3の120・180・240 Hzは背景構造があり、見かけの周波数依存に寄与し得る。"
        ],size=10.1,step=.046)
        savepage(pdf,fig)
        for start in range(0,len(pos),20):
            g=pos.iloc[start:start+20]
            fig=page(pdf,"課題2　位置ごとの推定値",f"微分Voigt型　　{start+1}～{start+len(g)} / 78座標　　± は標準不確かさ")
            cells=[]
            for r in g.itertuples():
                cells.append([str(r.date)[4:6]+"/"+str(r.date)[6:],f"{r.x_cm:.1f}, {r.z_cm:.1f}",
                              paired(1000*(r.f0_MHz-16),1000*r.f0_MHz_sd,3),paired(r.field_mT,r.field_mT_sd,4),
                              paired(r.fwhm_kHz,r.fwhm_kHz_sd,2),paired(r.height,r.height_sd,3),compact_flag(r)])
            table_at(fig,["日付","x, z (cm)","中心 − 16 MHz (kHz)","磁束密度 (mT)","吸収線幅 (kHz)","山谷差（記録単位）","注意"],
                     cells,[.045,.79-.032*(len(g)+1),.91,.032*(len(g)+1)],8.9,[.08,.11,.19,.18,.15,.18,.11])
            lines(fig,.085,["弱＝弱信号　 相＝残差相関　 短＝背景範囲不足　 限＝形状量が下限付近。元の解析の警告を含む。"],size=8.7)
            savepage(pdf,fig)
        for freq in [False,True]:
            g=mod[mod.voltage_V.eq(1.3) if freq else mod.modulation_Hz.eq(280)].sort_values("modulation_Hz" if freq else "voltage_V")
            fig=page(pdf,"課題3　"+("変調周波数ごとの推定値" if freq else "変調電圧ごとの推定値"),
                     "有限変調Voigt型　　"+("1.3 V固定・定量採用18条件" if freq else "280 Hz固定・10条件")+"　　± は標準不確かさ")
            cells=[]
            for r in g.itertuples():
                cells.append([f"{r.modulation_Hz:.0f}" if freq else f"{r.voltage_V:.1f}",
                              paired(1000*(r.f0_MHz-16),1000*r.f0_MHz_sd,3),paired(r.fwhm_kHz,r.fwhm_kHz_sd,2),
                              paired(r.pp_kHz,r.pp_kHz_sd,2),paired(r.height,r.height_sd,3),compact_flag(r)])
            table_at(fig,["周波数 (Hz)" if freq else "電圧 (V)","中心 − 16 MHz (kHz)","吸収線幅 (kHz)","山谷間隔 (kHz)","山谷差（記録単位）","注意"],
                     cells,[.05,.79-.033*(len(g)+1),.90,.033*(len(g)+1)],9.3)
            lines(fig,.115,["弱＝弱信号　 相＝残差相関　 背＝背景構造　 幅＝線幅の相対標準不確かさ25%以上　 限＝形状量が下限付近。",
                            "1.3 V・280 Hzは両系列に共通。周波数系列の20・60 Hzは参考条件。"],size=8.8,step=.034)
            savepage(pdf,fig)
    notes=["# Voigt解析の図表","", "課題1・2の主モデルは微分Voigt型、課題3は有限変調Voigt型。全条件で直線背景を使用。"]
    for (filename,_),caption in zip(FIGS,CAPTIONS):
        notes.extend(["",f"## {filename[:-4]}","",caption])
    notes.extend(["","## 数値表","",
        "課題別の主解析表は測定値と標準不確かさを物理量ごとに併記する。全105件の詳細CSVの _sd は再推定値の標準偏差、_p16 と _p84 は16・84パーセンタイル。磁場の換算係数は42.5759 MHz/T。線幅は変調前の吸収線の半値全幅、山谷間隔と山谷差は背景を差し引いた観測基本波のフィット曲線から算出。課題2の変調片振幅0は微小変調近似の指定。",
        "", "4モデル全420件の full_rms は全点フィットの残差RMS、held_rms は基準中心の前後6 kHzを2 kHzずつ除外して検証した予測RMS。両方とも記録単位。",
        "", "課題1の不確かさは既存の300回の再フィット。課題2・3は各120回のブロック・ワイルド・ブートストラップで、中心化残差に自由度補正を適用し、ランダムに始点をずらした連続ブロックごとに符号を変えた。全件の有効回数、ブロック長は詳細CSVに収録。±は標準偏差で、推定幅が下限に近いときには百分位点も参照する。",
        "", "注意を要する波形.csv は元の解析での警告、既知の背景構造、形状量の境界解、線幅の相対標準不確かさが25%以上の条件を記載した。25%は図表上の目印で、採否の閾値ではない。課題2の範囲外・個別保留、課題3の20・60 Hzは参考点と除外点.csvに記載した。"])
    (dest/"図注と表の読み方.md").write_text("\n".join(notes)+"\n",encoding="utf-8")


if __name__=="__main__":
    build()
