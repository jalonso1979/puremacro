"""Replication cases — econometric regression family (pure-NumPy estimators).

Replicates canonical published econometric findings:
1. Card (1995): Instrumental variables vs OLS estimates of the return to education using
   geographic proximity to college as an instrument (2SLS beta ~ 0.132 vs OLS beta ~ 0.074).
2. Long & Ervin (2000): Monotonic standard error hierarchy under heteroskedasticity and leverage
   (SE_OLS < SE_HC0 < SE_HC1 < SE_HC2 < SE_HC3).
3. Mroz (1987): Female labor force participation binary Logit model with Newton-Raphson scoring.
4. Romer & Romer (2010): Macroeconomic effects of tax changes; narrative tax multiplier OLS with
   HAC standard errors on vendored quarterly US fiscal data.
"""
from __future__ import annotations

import base64
import zlib

import numpy as np
import pandas as pd

from ._model import ReplicationCase, TargetKind, Tol

# ---------------------------------------------------------------------------
# Card (1995) closed-form moment Cholesky factor
# Variables: [1, educ, nearc4, exper, expersq, black, south, smsa, lwage]
# N = 3010 observations from NLS Young Men Cohort
# ---------------------------------------------------------------------------
_CARD_CHOLESKY = [
    [54.86346689738081, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [727.6791325395795, 146.84032166419772, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [37.42016529578831, 3.6851361529332656, 25.28143589217639, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [485.87888275198685, -148.3442944920825, 7.475873491044303, 171.90925775790325, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5243.799130268498, -2931.321570582817, 127.06908249874547, 3398.054078792225, 1178.8364844657997, 0.0, 0.0, 0.0, 0.0],
    [12.813626986331798, -6.253111949723719, -0.8482299440705421, -1.0886149319063483, -0.0059480494189404895, 22.31154280887121, 0.0, 0.0, 0.0],
    [22.14588447851086, -5.421521811053248, -5.249844682222076, -0.47828873492912277, 1.038778789226723, 7.814876213134012, 24.601339821557584, 0.0, 0.0],
    [39.11528237932865, 4.655568997817778, 8.193492768021747, -0.7906271000897253, -0.7619885135524092, 0.6269356395202064, -2.4174994936541716, 22.79789394106864, 0.0],
    [343.5458101943803, 7.6495339965184055, 2.9102822702790982, 6.876034755512275, -2.8856430977851226, -5.090070771391767, -3.375604653353329, 3.533504388962253, 20.50025758231625],
]


def _eval_card1995_iv_vs_ols() -> dict[str, float]:
    """Replicate Card (1995) return to schooling: OLS vs 2SLS."""
    from puremacro.regress.ols import ols

    n = 3010
    L = np.array(_CARD_CHOLESKY)
    rng = np.random.default_rng(42)
    Q, _ = np.linalg.qr(rng.standard_normal((n, 9)))
    D_t = Q @ L.T

    y = D_t[:, 8]
    # OLS: lwage on const, educ, exper, expersq, black, south, smsa
    X_ols = np.column_stack([D_t[:, 0], D_t[:, 1], D_t[:, 3:8]])
    res_ols = ols(y, X_ols)

    # 2SLS: instrument educ with nearc4
    Z = np.column_stack([D_t[:, 0], D_t[:, 2], D_t[:, 3:8]])
    res_fs = ols(D_t[:, 1], Z)
    educ_hat = res_fs.fittedvalues
    X_2sls = np.column_stack([D_t[:, 0], educ_hat, D_t[:, 3:8]])
    res_2sls = ols(y, X_2sls)

    return {
        "beta_ols": float(res_ols.params[1]),
        "beta_2sls": float(res_2sls.params[1]),
    }


def _eval_long_ervin2000_hc_hierarchy() -> dict[str, float]:
    """Replicate Long & Ervin (2000) HC0-HC3 standard error hierarchy."""
    from puremacro.regress.ols import add_constant, ols

    rng = np.random.default_rng(123)
    n = 100
    x = rng.uniform(1.0, 10.0, size=n)
    x[-1] = 20.0  # leverage point
    X = add_constant(x)
    u = rng.normal(0, 0.5 * (x - x.mean()) ** 2, size=n)
    y = 1.0 + 2.0 * x + u

    m_ols = ols(y, X, cov_type="nonrobust")
    m_hc0 = ols(y, X, cov_type="HC0")
    m_hc1 = ols(y, X, cov_type="HC1")
    m_hc2 = ols(y, X, cov_type="HC2")
    m_hc3 = ols(y, X, cov_type="HC3")

    se_ols = float(m_ols.bse[1])
    se_hc0 = float(m_hc0.bse[1])
    se_hc1 = float(m_hc1.bse[1])
    se_hc2 = float(m_hc2.bse[1])
    se_hc3 = float(m_hc3.bse[1])

    return {
        "diff_hc0_ols": se_hc0 - se_ols,
        "diff_hc1_hc0": se_hc1 - se_hc0,
        "diff_hc2_hc1": se_hc2 - se_hc1,
        "diff_hc3_hc2": se_hc3 - se_hc2,
    }


# ---------------------------------------------------------------------------
# Mroz (1987) compressed dataset (753 obs x 8 cols)
# Columns: [inlf, nwifeinc, educ, exper, expersq, age, kidslt6, kidsge6]
# ---------------------------------------------------------------------------
_MROZ_DATA_B85 = (
    "c-n<r3wRXexgIb|j8RuzHEN_0BL+ncPO7N{nH^S0kHH3{7!`RU;D8!!IC5G9F9{kAh8V(qgI5CL1wjr-(Sjm"
    "Ki@H&yT<sBz6_KJ9QEsQgc6a9gcISO}CY|TmhX+3X|Ns8)ew)#>n!a<Jt_*3K5&u=jf8l<brkPs&-!=QG>3b+"
    "PSHWK}G;Krt&;20>rWpzNgRO>wSI2*M#@A=7^@a^^j_fcT_?q~ifjBJFRLgDnzUT|;>@VVf5659gLh5`QzT}+?)"
    "&3c#mRBFvv<kJ}!SBQS&rJOP_^&YjI}nEjO*ei|TXMfRUU`wU_4BKrQt&nLUmy-QV0~NkCIx>vq}BXi{C>Frzs"
    "Xul_s=k8@5Jxj$g>JJp>F}eGwpRoD23C0xPMFJZsJc*IRC=z_3?ktbJ4da?BCFLiaI|K|IITU{-o$%7r9gT8~<"
    "%c+rOde;Slmq?*D?qPu8ytX+}nzenm*%Xug8;`8D?X(S6=j{1f>qfb+LjcT($xzxzY({f}>Mi2uDj{+pWEzn_b"
    "LHhkT|9O4h+_e(27!EJm(Dfx4L>!Y8=|6cFnKhnopV>s)K>iGSuVE@ChC+Yqg(0_AetlB?K{_XYM^oeS{zzbmi"
    "x{q58hc>p}#?LPW_(|Fmq#uai;K;3teu@86)6d~?>V98G>q-8~K)!yvt4^)m9>4!JF8(|B+thueI$Iu;|JS-+q"
    "V|uUQ!_Qbe=*>9<0aQ9`TintemDhp)>qWtt#A|lEra!6Yf}_`z2)aX!Fb`n;yLQykFh0m|532NrY%?Z7yUr{wc^"
    "L;)%wI8HkfQ*CI&~4ehGdK{VuTe!Y!Yfu)cSnSJe65`)_*cE4BWO#QoFc|5m+D?yvZd^qbl5Bf}d%XaB>KK2z&"
    "$8SlsZ$<W4#eN*x|2KK)rFp2atX}=X66Z)&U<f9zzODtwMcaPJhW@?z>UA0nmH_^W%!2fPGUlYIhxp}X}*AIjB"
    "jn+#_euTc60ACTDtJaHu!S!8x_4l!#Vn4o8>*vJpw>*BoJkX!}qSYinJ;8o{AG=i1N93FBUkTB-1m$=1P6PRu5"
    "BnFb?63A0cmVCE-k|6s`cVw``#JUpvfoI*+EuTReawgZJ9gdh&~Hb+&akd0`OAR)J7sKDu!N24ktRRaTc$euo4"
    "EZNoF<<=^otek0e?g2Gjm@j+F#_$!SC9p%M@JrYXJV6qH;gkQ$zn7Ia$$H?jHvHwzb}-cnki;koCi?PA1A9-tVe"
    "mAq5lrfd0d>=~JoVzB}XRR|9->QzOY|Dcr9mvR>@7I=>Ru-#k6jK>DNoSr%ES?l15P*nfytplF}idtRFVZHpEt+"
    "6euNN&l`j9s1e!#$@}oHTO2+Z!W~Qx9)8w`&kJ4m#?fR`Ogsk*k_#8{)0#S=#Kw(ll?&Z-aG4AKlyX)^VI`G-u"
    "B<QpJly6`i1au)+V*Lw|w3mSwQ%qed(-ir1j{(24r4I@*jlrYj;P<{{{g5`@IXvev|<H4nNvxV*F#!{p)Ex#-G"
    "Py3l;uyzYL6@ZT&Cwk}vy=N#_WD2|Dx$r|9FXKX7X!*@r^F@0G|0V&9aW1i(Hw>>KG3Kc?iXEjmi_D-}PXe|kQ"
    "$TFI`+5Ar`2OXx>E(KoezPwa9N<+m8_x94B?D*8x%f%dChuT}RG{YLnuy`Coi6p{X}Q20swmIL}3w*FVyCed%i"
    "{~jyLbnY(nuK@g?+;^VBU-FxLSbt?QzZCvpey~0^n&^x9MW(Hn^}hJsp*K-{oDKUwX$k)%KE(LDHCp0lU+n$g>"
    "Q+nn0ovDmeY(O;<O}n+50d^}9*ln*XFo-7l)oLV5%M3IbiY@{ekl6n0DikBtdjgf*$<T8w&+Wg-<1Gd*FXIa@t"
    "MRnl3xaZe$8ta5dYDBU!~QP{YC$JJoYHnH_$(yuRSCBt@Ic5BO_@W)8x}WfA5>OtNkTE!}w`tWQWo%;r|fWe`4@"
    "eivI&}zf}G!>(Re%I=a)xemn3J?0C1*CEr5;zo)E6lza*t<Kz9Yzft^&_B+s{Udg`O|EBbBYr!(guY1D&Es<X8"
    "e33uQ&o@RpEBoW^{{~w-Nx!lFv^<(k`CSIsuSxGetIppL=U;BvWnVNWKmCRHC*{8}emPUWnfQbCpF_9YNd7Ab@J"
    "FrBD1XBGT1(^=%3rX)cduos_3rwWOMm9hC;dSGc_=nOoh|f5{KqEqdy#)!pJn^gN)P|!!29eOMEZ~RySKg8zy8#"
    "J@cCqaQ2)AL(Lnb@`})S5qXybP%pcb04^}XtUm?Y}QObWbjGwIGs<!6szY9mSDEf(iLiq{nXQ}&%{R_hSW3h9U{"
    "}=f#Bl}rsBK^X!|2IcpRd*BoFuu8<Vl~w-b78&n?{52o@o|kEKe_kM0eI8eS1A6>19-3YE~Z0&p|8aMPW^LOQ@x"
    "V!wEBgOWB9RzT$TIYe^au)CiVy8%lAG!f#M&Ozl+<3Q2dDY<&>mvm-BOhzHP0gA><$0zo5<cfJc5I_VK9H4_)>v"
    "1^+a*fbNI(yT5H;-S}gCKfii`PyNyPKPTGwm3!!ivwq~tPAdKv|6Bm)&x@R3B7O5<{l@4T#b3cc6Z}Wf=%*?EM*"
    "f_v%_sbkKQrz6jp3XB<{a-%`77p^w=^~T^amXLepfe;^3zO;kA^AwiT%O);Oxj!vM;#+-*?wm1M^3WZ!Xd{Q2c="
    "PmwAx~ed<5X{qE3jHqicIeqZ*(JqG4ixc|!ojvB~6wErhxF^%HGe9-Tjepe~o;r@<;-;vlY9riyAU01YC+AD~k"
    "HpUhb|1m#}MA|&^TUUSQyVJ;j;ra6hM96=k|GshMW{R(h!9HB0FAHIOgZ5!WV1ZwM!iGN^X_xrNmH(&iZ&_bc{h"
    "^%v%jv`)j887JuJO5_L!bY#x+~iz@e}U<Q|lZL|DK9Zz8-fj)i2OLKca0k(0*Zlvf%q~I*c!W-ZV$yC;Ov+cqcl"
    "E`g52+J$IgFVtpF@+uM;kMSsB$?|<S?5d-r}tRJ7zdA+(n#~052*Xa#JKa4-7TID34h~L1|=1~5J^@H{amBfGa9"
    "~svDN;lH@<Mc=Eo?@QMf1!PxpkGe?-7>0Qt)=)E<>Th2Toe6YIn@WgHjuxv-mx#=J#<{{FZ{vtuTJ9b?oX!Zw|Kw"
    ";vJaV{zk6)^<nBLX{@6R~BdTv<|Nc^a8tLZ{h~F>#EZ4;T8TL0<+y2c>{~Ulf-*<|M`7`>*oQ$ui|AO{!YwS15H"
    "j975{C`s95)=7{>)R((slA2%=wCjHO71B1NBMp_@*T-9o_}aiEzM8l0e>!={{hWElz@I*t&a?0{DS_Y`G#h-K2F"
    "z~<zcR`7w_03v7Mq%KH8@nhN`oLKL+sU%jlWZAI0;Z8MK}38`j5)tid77zYt!bKSuNmpntLJKVJ1=2fldj<dC=h"
    "b@tzRFjJkMmOnfB@A6L!%&$@YcG~%;`+jKOPt~?leGToysoGGV{73blV}q3akn>S~W<<`T`Xt(iF*gjC_=5YxHv"
    "FN;g9gTjSYL}qGfku)%E!8h^cN)mL;tiVHr-=>$Xy@4O2ODaIPh!qE2Mr)`sLvFXmFOgpTtL)zm9B)D81nM90$K"
    "n+9I-FNdJ!peL?br^5yg|YJBHAv?TwfK8Nw|y1OJl5cxp+an-dARG-86^z#XvpQHTlv+F;FzVVUXel^VpAbqp{H"
    "cRRkD*r?JeHjgt|3Uegu;n(Y58(QD4@&=D>=)LjkHpqdev0(d^|vTL#r(`1_n^9)+;2a`XBVup6#b;Wit`%_TN{"
    "Y~h<|%Dm*&T?K3Y3x3e5)@l)pYs_8sdh*RC8w`i1g+a%~@qe`vmI#&ss<XSlvVznAhq^nVYn+(`OcLjB*aRNqGV"
    "{Nwup<y*x+VEoZdJEV9c@{RH3d$BH*Un2j0jxDA55$o66S}&*ky&UG#bp1rr+kZIndvn)2Xn)j?Pot|s*#E-%^?"
    "+n-EcC_xOuJqGsP~<}X^DJ5`78SGUDqEqaQ*=6Q@^gel;ZCqs?YO$Cf;x7)+ecdTtxcUMD)S)XSR-~`~>~u$N62"
    "Ue-#G#SQh<?>gy<<%Lg<n+`Q*Ew!}VBFu5PrH@3WZE6H~u&38UT^=XVh?y~Z!zK`~4n$^>%f9}A0*>Y3qvQKz^e"
    "y?>D-;BciTtA!oui3Ew?$|D>KNORHy<MFT`X#-azK36*O!gJ)W2YzAde@KgPduIS&m!ud|B32n*&v^v$6}QK<RS"
    "g^$v*Qj5`WE8v=RDX{8rewgyJ8}KbAx`Q2vDZ`R~^@%iY!eE2w|>3EB5tSbvt5Me{d7nm_38*Z*+(TeEsn{TuB="
    "-ONe<VSef-l5dPJKfQXO;-Baz)=zut8451)gY!KnXs4+Dis)w$>~Co@z8Cqz{nyy>Wu<R?@+<wXM!Ni^>aRZy3Q"
    "&Cw*Z<D0|9H{gq0g>CV<^5s{N_X^lYPph{zgx;qyF8LAMa4IA@&L7Yxl|jT$QdLsr|PXJfZ00UEhsFjvC(j>HMF"
    "5_WWg@%RZugYm5GDp#8x5>)gomCh8~VZ&O~}ZD9U^{kegsP4VbYrRdWZo$qCzlKa0oVTOWv_dnaBLB(6|`uh3XS"
    "{3aDj`S@^&Zm0EpO*C;)%Q{U%B*QLAAtVp`&bvM4<UWJWM8N7_ntpJX3uZ9=SR@~S6aCy$_M6WA5>3Qychk>f%!"
    "l?Znf8X&)=(j$7O$ce(FQDULMR(zd!a?hx0LSRsDtPZz$g<Jyz9Wd|~RdJDi_#?&sFe0LWifvbX15KiE9sA2eT!"
    "@$s}igF_wV*S}f0f%^Y>;Qt0E-|z6AADthG^u_oB{d*@{pDL+7lajAx2Ss)So(tz^Ynv5*JYS~jyIPsxPWfexs^"
    "9y)ALGE!v+JvcRNqO_|Dzu)x_=nX&s?<HWBx0pf6pb~pMd&9a{jF`(k$~|N`3?2pT5j{lkms<^2@HTE8N6C_oV*"
    "MyTtztdcUWJ>PPPUM$>ftWr}ay`m1SecK%Vr?_U`r0hYDGBY%?mlcsHJJ)P>a?)h#_Yl*y0>x<z0+L?=e^3NBH)"
    "Vx=@$NZ$=KaTk4o*z!YZzBIZ3gC+eJgoR9_jmUX68Qf>_0>w+|8a^x-1X_iQ>b|3O@A%<ewN@j4EC@1)qLqcs3+"
    "i(Pfa^7V=&pjLO8!II$qvyQD+yz`4fXv#XqX`?)W(#-5s0dm0uXLzh(WI&M%_-O(FTqG~M!}+w+<A9{GdwTyoL_"
    "inbD8<kJ3Y4CGG$_@`-SQ~eM5J2AME>cc~T{`0HP@TgC__aEcI{}g{(?D#X_@-Jn;pT7=#g6eBIaQ>l!8D8@nN&"
    "2)z783pyu>P0B4=BC}e{$gbIrmHMpQistN)OZKCzJeNJfPh``OGxY{#;D`mr|ggWz|ssh3!uU)mM;z4OLxClux|"
    "BmYk0e`nuo0O6bQ|9n6m=o>^4?D}(!4);zKgs9%Fu4yFB3zI1#3HP7SyH3z>{<Ldn42OB<ib!QX%$L{&$gnYDny"
    "nkcMrw#AD>k1{if`2y1N7LG7x?cv^hx_(yR`yxo2GMT=$!7`Oe<IN*lkERRrpNq`k&>_J3NG?>B$UccHEqlI9@P"
    "IHL-=W=-(i4T)=*hb@*(=YI)w32A>D5_)h`(yGV%Ss0MUOP`A^i}K3Tiz{lFl=hc$iZGvDDn?Gq||>a*fMswjR!"
    "`=%%R^X~bslzapiKkD&*W=j5En)^7>pZ)9N5b`$%<fA1mzAXp+Xp1fi`Pd)ztfBln1K=Y93khCG_Wv2GujiBfyP"
    "WJ7`^WAk`d_pUgICU`{Amc$r-uA%0iAy@&0lFI+SgfXy}W-_N%W}_|D^a61pAW8f871&6#rWyr_%dR6(C=_9lw8"
    "J_}NEId!OvL0r%@Mte);4fb-MuH;VSJr2M}O;Qd!Lll)~seEaA<FH-(kPW;J`_)@_uN&iQZ{X_ge&tFXa?Eu724"
    "OQRz%(o=<lj<k#`AU)hLf8Gu$^K3A*<biO+3WrLBtLh2{20YARX+BW-~V**U-6@EqI{H*f8+Y;5y(%RzG<b;d`p"
    "5q#e^TnZ-4KgzN~52P9974GY{nZ^L&l+L-e29S|v6Y`M~(4B~lkkk8e}>S=LDupOlh)k#}3X_itL;MESEp{!QvT"
    "qQ98mDSzux->OmZ^?QE(J;lFcG#`Zat54R&;y>y83C{Y)*^9_NaQwc7<g<+K*XR?Ui2m&J@gEL+-VI9??q2T~CZ"
    "24v{}^9S3}%u3WBjqLwVwQIIr*Q?rqBD4Qlp^$T$t?BEP@vjf2XLkg?>oiPrJ<}{IEWt>!Zm(R#5!ESjoT8w}j;"
    "TkgBf=e=5m-9H;yl^Xn0TrxZ-&ALG-TMhqqV!&INBA^ulV{`@w{UoqMDvrV7=r>j0-_{^sy)(@ioW>1R0ZV#n@z"
    "aa&0i*E4w{)@n8(ELaa*l&9-2HxMxfc-0XN2ouD{v&JKa}+=HB>#V|PktdXw4eGfMdZJ$seTiH{NSaz>xqAvBwv"
    "qE{UZqUuiIBk^Lxcqe|nkxe+l8YlIm+mLT>y@+dni?xlZEY_@oEr&&a<Q&c8R54o^K9Lti8RhV`MA$csug1<v*}"
    "6iR(7Bk{xdXkzfs9lXCT_jl({C?9qES_ocD^~0rx_xy=Nvv1F+H+}Y3_T?WWU#x#KlzhhjJ?~hU>xcUnE~of8pV"
    "lv;{4$sFuP=x{Ve&7(R`IdBzv!0#RX+M5^c_$BKLGUWlXVOA9~1Iz7%BdD@~9uX>m#|;A1);Q;rfT+QQs`|n6Iu"
    ";{HZkY{Uy|&XWm~y{yRYV-BO?Z#lJREeo##K&FIj6jqNkkr@kup|8MGl7EydrEBTGfKc@6QxVQ`HcOLP#(x*S^t"
    "RM3CI-mN!_>Y5R|BE62cIE>LedhNy?X)ZA(EX2q{du^rPW9yikdN%n&y)Snp!)bInx97d_wAr*YJb)riC<ro{-C"
    "075$V@bsy|@-u;*Vtk^JTZ{U-*`Q84fP&FR~?@iRz1<o$WEUud5$SaOny=c9e?z3T>|UpCqACB$F$kD7`4jq!hJ"
    "`*O117@ut!AM)Y9^9+A`tfAsx%HM^*0af1+`OG%m@vp9L^T~gm^`9KpD1NRY|C6ocN9a>Q@!>fp#`i(8uWR}uyp"
    "Zcd^!<lS$lr9m*~h-iGoHRr751n<xa;TWf2QYDk$o=nkU#!@V(Naw-{$$R60#5LLf-w88drV5@aPY@{g2x|jUoC"
    "UqW(SV*D(A0MQ;7!{Oq?K){o~`U!>^go)2@^e}<blKUGcnb(=^3(=8w0(EEAVAK2S{uaW+KO6vUV&gYsP^{4iBA"
    "4L8Q_kUoF)bG9KccuQ}KEH(g+jg>lL9$N=D1R!Y_{Q+~ew|ytf2Q~E(Ei=CYCHMIp0NKP-dU*d6ZtVrtlw;>`g%"
    "Uq7j?29rIg>EL-FAlqF=kbf2W=t>OW-C`{zN5{~uEJ$LsqWb_8i)f29EQYw>^vl3%W`mXUqK_;dfln}|PH-_rFZ"
    "KJ(2gI`7)(GryAT|0&+d`U;Qy%fEiH?qDVLKMcxW9}Z!EvVh{Z$EiM%Px#5VC1w7e`*(Tt{$7EH{jBzw?-Tv_r_"
    "cBC<b1wA#^;v(eV}TO?+1wdjG_DC{#o0erTn~v<RjbcsQ<et_vH@e+Y|ExSE{vQU&;u-oizW9^@r5=Po=&-gzWz"
    "TvY(iL_sKe}^uzsq3U_^?far_y*SR~A{Tu3^3Vwg=u)dO-ACL$ATU&H(sH6OQ)wmxGtnZh3*cbQu5~_dHlKxf@{"
    "mWE-=&m0aMk>GmFN(jp{xFBWk5@wR*&LtmQ;7Upl%He$tf6WT@lX1<PR%~tu*B#6Tsi-GGo3#v`s?~0>c4S(-QT"
    "DGCi5#BReeS1gZYtJxs2vZs)&C}sXoH};}-G{`E<WG2#)oKZwGCp`V+?=kC1-nQ+yQi`hG`JKKf)$@p!-19Un}k"
    "{_+v1AC)&s|3u`2?O!GF7xlOIu1>_CGRp6+SGc?Dn=rv*`hUMWq5nseeM$R%p_cNqhOd3mv~Ffc^Be09E()b)xW"
    "Z}j_rRDxQU8PKyOF-HQvmW&+W!AEzm%);qctkN*<oP+2;<9PP4fLLxBqtcSK4VlBair(NBt|QPdfOG>ir9!`8>%"
    "_|3dx?<LmqOd>Kl8L&yDn55xF39$bEJ"
)


def _eval_mroz1987_logit() -> dict[str, float]:
    """Replicate Mroz (1987) female labor supply binary Logit."""
    from puremacro.regress.discrete import logit
    from puremacro.regress.ols import add_constant

    raw = zlib.decompress(base64.b85decode(_MROZ_DATA_B85))
    data = np.frombuffer(raw, dtype=np.float32).reshape(-1, 8)
    y = data[:, 0]
    X = add_constant(data[:, 1:])
    res = logit(y, X)
    return {
        "llf": float(res.llf),
        "beta_const": float(res.params[0]),
        "beta_kidslt6": float(res.params[6]),
        "beta_nwifeinc": float(res.params[1]),
    }


def _eval_romer_romer_tax_multiplier() -> dict[str, float]:
    """Replicate Romer & Romer (2010) narrative tax multiplier OLS at 2-year horizon."""
    from puremacro.regress.ols import add_constant, ols
    from ._data import load_csv

    fiscal = load_csv("tax14_us_fiscal")
    try:
        shocks = load_csv("tax_shocks_rr2010")
    except Exception:
        shocks = load_csv("tax14_narrative_tax_shocks")

    fiscal["date"] = pd.to_datetime(fiscal["date"])
    shocks["date"] = pd.PeriodIndex(shocks["date"], freq="Q").to_timestamp()
    d = pd.merge(fiscal, shocks, on="date", how="inner")
    d = d[(d["date"] >= "1950-01-01") & (d["date"] <= "2006-12-31")].copy()
    d["y"] = 100.0 * np.log(d["gdpc1"])
    d["dy"] = d["y"].diff()
    d["rr"] = d["rr_exog"].fillna(0.0)

    h = 8
    d["dy_h"] = d["y"].shift(-h) - d["y"].shift(1)
    reg_df = d[["dy_h", "rr", "dy"]].copy()
    for lag in range(1, 5):
        reg_df[f"dy_lag{lag}"] = d["dy"].shift(lag)
        reg_df[f"rr_lag{lag}"] = d["rr"].shift(lag)

    reg_df = reg_df.dropna()
    cols = ["rr"] + [f"dy_lag{lag}" for lag in range(1, 5)] + [f"rr_lag{lag}" for lag in range(1, 5)]
    X = add_constant(reg_df[cols].to_numpy())
    y_reg = reg_df["dy_h"].to_numpy()

    res = ols(y_reg, X, cov_type="HAC", cov_kwds={"maxlags": 4})
    return {"multiplier_h8": float(res.params[1])}


CASES: list[ReplicationCase] = [
    ReplicationCase(
        id="regression.card1995_iv_vs_ols",
        family="regression",
        paper="Card (1995), 'Using Geographic Variation in College Proximity to Estimate the Return to Schooling', in Aspects of Labour Market Behaviour, pp. 201-222",
        title="Card (1995): return to schooling OLS vs 2SLS with college proximity IV",
        title_es="Card (1995): retorno a la educación MCO vs MC2E con proximidad a la universidad como VI",
        source="model:closed_form",
        estimate=_eval_card1995_iv_vs_ols,
        target={"beta_ols": 0.074, "beta_2sls": 0.132},
        target_kind=TargetKind.POINT,
        tol=Tol.TIGHT,
        citation="Card (1995), Table 3: OLS educ=0.074 vs IV educ=0.132.",
    ),
    ReplicationCase(
        id="regression.long_ervin2000_hc_hierarchy",
        family="regression",
        paper="Long & Ervin (2000), 'Using Heteroscedasticity Consistent Standard Errors in the Linear Regression Model', Am. Stat. 54(3):217-224",
        title="Long & Ervin (2000): monotonic standard error hierarchy OLS < HC0 < HC1 < HC2 < HC3",
        title_es="Long & Ervin (2000): jerarquía monótona de errores estándar MCO < HC0 < HC1 < HC2 < HC3",
        source="model",
        estimate=_eval_long_ervin2000_hc_hierarchy,
        target={"diff_hc0_ols": +1, "diff_hc1_hc0": +1, "diff_hc2_hc1": +1, "diff_hc3_hc2": +1},
        target_kind=TargetKind.SIGN,
        tol=Tol.COARSE,
        citation="Long & Ervin (2000), §2: small-sample leverage penalties widen standard errors monotonically.",
    ),
    ReplicationCase(
        id="regression.mroz1987_logit_participation",
        family="regression",
        paper="Mroz (1987), 'The Sensitivity of an Empirical Model of Married Women Hours of Work', Econometrica 55(4):765-799",
        title="Mroz (1987): female labor force participation binary Logit model",
        title_es="Mroz (1987): modelo Logit binario de participación laboral femenina",
        source="model:vendored",
        estimate=_eval_mroz1987_logit,
        target={"llf": -401.77, "beta_const": 0.425, "beta_kidslt6": -1.443, "beta_nwifeinc": -0.021},
        target_kind=TargetKind.POINT,
        tol=Tol.TIGHT,
        citation="Mroz (1987), Table VI & Wooldridge (2010), Example 15.1.",
    ),
    ReplicationCase(
        id="regression.romer_romer_tax_multiplier_ols",
        family="regression",
        paper="Romer & Romer (2010), 'The Macroeconomic Effects of Tax Changes', AER 100(3):763-801",
        title="Romer & Romer (2010): narrative tax multiplier OLS at 2-year horizon",
        title_es="Romer & Romer (2010): multiplicador tributario narrativo MCO a horizonte de 2 años",
        source="snapshot:tax14_narrative_tax_shocks.csv",
        estimate=_eval_romer_romer_tax_multiplier,
        target={"multiplier_h8": -3.0},
        target_kind=TargetKind.POINT,
        tol=Tol.MEDIUM,
        citation="Romer & Romer (2010), Figure 4 / Table 1: cumulative multiplier near -3.0 at 8-10 quarters.",
    ),
]
