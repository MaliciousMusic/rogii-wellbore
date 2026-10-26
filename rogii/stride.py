"""rogii.stride — STRIDE backbone: a joint posterior over a SEQUENCE OF SEGMENTS.

Implements the architecture specified in `docs/20_stride_spec.md` (Shrey Gandhi's
"STRIDE: A Joint Posterior over Depth and Distance", reformulated into build
instructions), plus the pseudo-typewell blend from `docs/21_corpus_mining.md` §3.

    state  = (h_len_1..n, dTVT_1..n)          -- NOT per-row TVT
    P(state | GR_lat, GR_tw)
        ~ p(h_len_1, dTVT_1) * PROD p(h_len_i, dTVT_i | h_len_{i-1}, dTVT_{i-1})
          * PROD L(GR_lat,i | prefix, GR_tw)

Everything here is INTRA-WELL and TEST-LEGAL:
    reads   w.md, w.gr, w.gr_obs, w.tvt_in, w.fe, w.tw_tvt, w.tw_gr
    NEVER reads w.tvt, w.tops, w.tw_geol, w.datum, and never sees another well.
(The one exception is the deliberate typewell-SHUFFLE control, which takes the
donor curve as an explicit argument -- see `shuffled_typewell`.)

No learning. Pure numpy.

------------------------------------------------------------------------------
DEVIATIONS FROM THE SPEC, ALL DELIBERATE, ALL FLAGGED  (see module docstring of
tools/eval_stride.py for the sweep list):

D1. LATTICE IS (h_len, RATE), not (h_len, dTVT).  Spec U2 calls the absolute-dTVT
    reading preferred and the rate reading the alternative.  We use the rate
    reading because it is the only one that makes the O(1)-per-extension
    diagonal cost tabulation tractable (see `_segment_costs`), and because it
    makes the clipped final segment (U25) trivially well defined.  With
    `rate_step=0.0015` the implied dTVT resolution is 0.15 ft at h_len=100 and
    0.6 ft at h_len=400 -- finer than the absolute grid U4 proposes at short
    lengths, coarser at long ones.  `params.lattice_abs_dtvt` is not implemented;
    switching would cost a rewrite of the cost table, not a parameter change.

D2. THE BEAM IS STAGE-SYNCHRONISED.  All prefixes competing in one pruning step
    have covered exactly the same MD.  This is stronger than the spec's beam and
    it STRUCTURALLY REMOVES the pathology of spec §3.2 / U7 ("a longer h_len
    accumulates more cost, so the search is biased toward short segments"):
    with tau=0 every lateral row is counted exactly once in every surviving
    hypothesis, so there is no length bias left to correct.  `tau` is still
    exposed, and its honest default is now 0.

D3. PREFIX DE-DUPLICATION IS ON  (spec U24, flagged there as OURS, not the
    author's).  Key = (stage, tvt_bin, rate_bin).  The state really is Markov in
    that triple, so this is an exact DP collapse, not an approximation.

D4. STATE VARIABLE IS TVT, NOT u = TVT + Z.  Measured on 40 train wells: over the
    toe, ptp(TVT) = 25 ft median while ptp(TVT+Z) = 156 ft.  The stratigraphic
    surface follows the wellbore (the well is geosteered to stay in zone), so
    "drilled-trend reading B" (dTVT = -dZ when dS = 0) is empirically FALSE here
    and would inject 150 ft of wellbore wiggle into a 25 ft signal.  Spec U20
    reading A (heel TVT-rate carried forward) is what is implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

__all__ = ["StrideParams", "predict_well", "decode_well", "StrideResult",
           "shuffled_typewell"]


# =============================================================================
# parameters
# =============================================================================

@dataclass
class StrideParams:
    """Every knob is here.  `src` tags in the comments:
       SPEC  = stated in docs/20_stride_spec.md as a source fact
       U<nn> = spec §11 UNKNOWN register entry -> a real sweep candidate
       OURS  = our own construction, no source at all
    """

    # ---- lattice -------------------------------------------------------- U3/U4
    stage_ft: float = 50.0            # OURS: MD quantum for segment boundaries
    hlen_ft: tuple = (50., 100., 150., 200., 300., 400.)   # U3 (spec proposes 50-500 step 25)
    rate_max: float = 0.060           # U4: |dTVT/dMD| bound. 97.7% of true 200-ft rates are inside
    rate_step: float = 0.0015         # U4

    # ---- TVT discretisation (cost lookup only; the readout stays continuous) -- OURS
    tvt_halfrange: float = 45.0       # OURS: |TVT - TVT_0| max over toe is 19.6 median / 40.3 p95 / 53.6 max (40 wells)
    tvt_bin: float = 0.5              # OURS: matches the typewell's own 0.5 ft grid

    # ---- likelihood ------------------------------------------------------ §3
    lik_stride_ft: float = 10.0       # OURS: MD block-average pitch. 10 ft of MD ~ 0.2 ft of TVT, so nearly no TVT blur
    misfit: str = "cauchy"            # U27: "cauchy" | "l2"
    gamma: float = 12.0               # U5: Cauchy scale, GR units
    gamma_mode: str = "global"        # U6: "global" | "heel_mad"
    tau: float = 0.0                  # U7: per-segment cost *= n_finite**(-tau). See D2 -- default is now principled
    # OURS, NO SPEC ANALOGUE, and the single largest lever we found:
    # the per-sample Cauchy terms are summed as if independent, but the GR residual has lag-1
    # autocorrelation ~0.83 (corpus §4), so the effective sample size is ~6x smaller than the
    # nominal one.  Without this the likelihood out-shouts every prior by roughly that factor and
    # the decode fits GR noise: measured, the MAP path beats the TRUTH by 80-370 nats in 12/12
    # wells while a rigid shift-scan of the same reference still minimises AT the truth in 11/12.
    # lik_weight is the temperature that repairs the mismatch.  1.0 = the spec as written.
    lik_weight: float = 1.0
    # OURS, no spec analogue.  LENGTH NORMALISATION OF THE WHOLE LIKELIHOOD.
    # The likelihood is a SUM over the toe's GR samples, so its magnitude scales with the
    # length of the toe while the priors scale with the NUMBER OF SEGMENTS.  A 4000-ft toe
    # therefore runs a 2x louder likelihood than a 2000-ft one at the same lik_weight.
    # If lik_norm_ref > 0 the effective weight becomes lik_weight * lik_norm_ref / n_samples,
    # i.e. the likelihood is a per-well MEAN scaled to a reference of `lik_norm_ref` samples.
    # NOTE: within one well this is a constant rescale (the beam is stage-synchronised, D2,
    # so every competing hypothesis carries the same sample count) -- it changes the
    # likelihood/prior BALANCE ACROSS wells, never the ranking inside a single well.
    lik_norm_ref: float = 0.0
    oob_cost: float = 2.3026          # U9: per-sample cost when TVT leaves typewell coverage (= a 3*gamma Cauchy residual)

    # ---- persistence prior on the RATE ----------------------------------- §4.1 (2.30 ft lever)
    # DEFAULTS ARE MOMENT-MATCHED TO THE TRAIN TRUTH (120 wells, 200-ft windows), NOT guessed:
    #   sd(rate)=0.021, sd(rate_i - rate_{i-1})=0.0206  =>  AR(1) phi=0.52, sigma_inn=0.0179
    #   => mean_revert = 1-phi = 0.48 ;  sigma_p = sigma_inn*sqrt(200) = 0.25
    # ...but the moment-matched values (0.25 / 0.48) MEASURE WORSE than a tighter prior on BOTH
    # 12-well samples, so the shipped defaults are tighter than the truth's own second moments.
    # That gap is itself evidence for the over-confident likelihood (see lik_weight below).
    sigma_p: float = 0.10             # U11: cost = h_len**(2-p) * (r_i - centre)**2 / (2 sigma_p**2)
    pers_p: float = 1.0               # U12: variance exponent
    pers_kernel: str = "gauss"        # U11: "gauss" | "cauchy"
    mean_revert: float = 0.90         # U22: centre = (1-mr)*r_{i-1} + mr*anchor          <-- §6 drilled-trend centering
    init_sigma_mult: float = 1.2      # U16: segment 1 prior = the stationary spread (0.021*sqrt(200)=0.30)

    # ---- log-normal segment-length prior ---------------------------------- §4.2
    len_mu: float = 5.298             # U13: log(200 ft).  NOT fitted -- spec says the author fitted it, we guessed
    len_sigma: float = 0.60           # U13
    len_jacobian: bool = True         # U15

    # ---- typewell reference term ------------------------------------------ §5 (1.26 ft lever)
    calib: str = "affine"             # U17: "affine" | "offset" | "none"
    calib_lam: float = 1.0            # U18: cal(x) = (1-lam)*x + lam*(a*x+b).  lam=1 = plain OLS = honest baseline
    calib_robust: bool = False        # U19: IRLS-Huber instead of OLS

    # ---- pseudo-typewell (corpus §3, radiantallomancer; NOT in STRIDE) ------
    pseudo_w: float = 0.8             # measured interior optimum 0.0->11.02 / 0.5->10.31 / 0.8->9.93 / 1.0->10.19
    pseudo_smooth_ft: float = 3.0     # OURS: running-mean width in TVT for the heel curve and for the coverage mask

    # ---- drilled trend ------------------------------------------------------ §6 (0.93 ft lever)
    # MEASURED, 120 wells: the RAW heel TVT-rate is a WORSE point predictor of the first 200 ft of
    # toe rate than predicting zero (rmse 0.0319 vs 0.0232) because its spread is 0.037 against a
    # true 0.023 -- but it does carry signal (corr +0.52 at a 200-ft window).  So it must be SHRUNK.
    # Optimal shrink = corr * sd(true)/sd(trend) = 0.52*0.0232/0.0371 = 0.33.
    # Reading A of spec §6 used raw is REFUTED here; see eval_stride.py header.
    trend_window_ft: float = 200.0    # U21 (spec proposes 200-1000; longer is strictly worse here)
    trend_shrink: float = 0.33        # OURS: no spec analogue. 0 = centre on flat, 1 = raw reading A
    trend_robust: bool = True         # OURS: Theil-Sen-ish median-of-slopes; heel tvt_in is clean so this is cheap insurance

    # ---- search ------------------------------------------------------------ §7
    beam: int = 500                   # U23
    beam_overfetch: int = 4           # OURS: argpartition this*beam candidates before de-duplicating
    dedup: bool = True                # U24 / D3

    # ---- READ-OUT ------------------------------------------------------- OURS / corpus
    # STRIDE is a POSTERIOR; the decode above returns its MODE.  Under an RMSE metric the
    # Bayes-optimal read-out is the posterior MEAN, and the corpus reports a sibling model
    # moving 15.17 -> 11.36 by changing ONLY this.  Three read-outs:
    #   "map"       : argmin over complete paths                       (what we shipped)
    #   "beam_mean" : softmax(-cost/post_tau) over the top-`post_k` COMPLETE terminal paths,
    #                 weighted mean trajectory.  Path-consistent, approximate as a posterior
    #                 (the beam+dedup trellis is a forest: each surviving state keeps one
    #                 parent, so path mass is Viterbi-collapsed, not summed).
    #   "marginal"  : proper sum-product.  Forward alpha with LOG-SUM-EXP merging at the
    #                 dedup step (exact for the Markov state (tvt_bin, rate) -- D3), backward
    #                 beta over the same retained trellis, then per-MD posterior mean assembled
    #                 from SEGMENT (edge) marginals.  This is E[TVT(md)] under the model, not
    #                 a re-weighting of K point paths.
    readout: str = "map"              # "map" | "beam_mean" | "marginal"
    post_k: int = 64                  # beam_mean: how many terminal paths enter the mean
    post_tau: float = 1.0             # beam_mean/marginal: posterior temperature in NATS.
    #                                   1.0 = the model's own scale.  Costs separate hypotheses
    #                                   by 50-420 nats (M2), so tau=1 collapses onto the MAP.
    shrink_w: float = 0.0             # CONTROL (3): pred <- (1-w)*pred + w*t0 (the hold-last anchor)
    keep_paths: int = 0               # diagnostics: expose this many (cost, eval-row trajectory)
    post_dedup_parent: bool = True    # beam_mean: one finish per pre-terminal state (see decode)
    marg_min_mass: float = 0.10       # marginal: refuse the read-out if the beam-truncated
    #                                   sum-product holds less than this share of the posterior
    #                                   mass at some node (at large post_tau it holds ~0).

    def __post_init__(self):
        if self.readout not in ("map", "beam_mean", "marginal"):
            raise ValueError("readout")
        if self.misfit not in ("cauchy", "l2"):
            raise ValueError("misfit")
        if self.gamma_mode not in ("global", "heel_mad"):
            raise ValueError("gamma_mode")
        if self.calib not in ("affine", "offset", "none"):
            raise ValueError("calib")
        if self.pers_kernel not in ("gauss", "cauchy"):
            raise ValueError("pers_kernel")


# =============================================================================
# small numeric helpers
# =============================================================================

def _running_mean(y, k):
    """centred running mean over k samples, NaN-aware (NaN = no data)."""
    k = max(1, int(k) | 1)
    ok = np.isfinite(y).astype(np.float64)
    z = np.where(np.isfinite(y), y, 0.0)
    ker = np.ones(k)
    num = np.convolve(z, ker, mode="same")
    den = np.convolve(ok, ker, mode="same")
    out = np.full_like(z, np.nan)
    m = den > 0
    out[m] = num[m] / den[m]
    return out


def _fit_affine(x, y, robust=False):
    """y ~ a*x + b.  Returns (a, b).  Falls back to (1, 0) when degenerate."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20 or np.std(x[m]) < 1e-6:
        return 1.0, 0.0
    x, y = x[m], y[m]
    w = np.ones_like(x)
    a, b = 1.0, 0.0
    for _ in range(8 if robust else 1):
        sw = w.sum()
        mx = (w * x).sum() / sw
        my = (w * y).sum() / sw
        sxx = (w * (x - mx) ** 2).sum()
        if sxx < 1e-9:
            return 1.0, 0.0
        a = (w * (x - mx) * (y - my)).sum() / sxx
        b = my - a * mx
        if not robust:
            break
        r = y - (a * x + b)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        c = 1.345 * s
        w = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-9))
    if not np.isfinite(a) or not np.isfinite(b):
        return 1.0, 0.0
    return float(a), float(b)


def _robust_slope(x, y):
    """median-of-pairwise-slopes over a coarse subsample (Theil-Sen, cheap)."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 8:
        return 0.0
    k = max(1, len(x) // 120)
    x, y = x[::k], y[::k]
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-6
    if not ok.any():
        return 0.0
    return float(np.median((y[j] - y[i])[ok] / dx[ok]))


# =============================================================================
# per-well preprocessing:  the reference GR curve  (spec §5 + corpus §3)
# =============================================================================

def _reference_curve(w, p: StrideParams, tw=None):
    """Build the recalibrated (and optionally pseudo-typewell-blended) GR-vs-TVT
    reference for THIS well.  Returns (tvt_grid, gr_ref, diag).

    Order is load-bearing (corpus §4, measured twice): affine-calibrate the
    typewell onto this well's heel FIRST, then blend the heel samples in.
    Reversing blends two curves on different GR scales.
    The same affine fit must NOT be pushed into the likelihood (10.35 -> 14.13).
    """
    tw_tvt, tw_gr = (w.tw_tvt, w.tw_gr) if tw is None else tw
    tw_tvt = np.asarray(tw_tvt, float)
    tw_gr = np.asarray(tw_gr, float)

    fe = w.fe
    hm = np.zeros(w.n, bool)
    hm[:fe] = True
    hm &= np.isfinite(w.tvt_in) & w.gr_obs
    ht = w.tvt_in[hm]
    hg = w.gr[hm]

    diag = dict(n_heel=int(hm.sum()), a=1.0, b=0.0)

    # --- §5 typewell reference term: nudge the typewell onto this well's GR level
    if p.calib != "none" and len(ht) >= 20:
        x = np.interp(ht, tw_tvt, tw_gr, left=np.nan, right=np.nan)
        if p.calib == "offset":
            ok = np.isfinite(x)
            a_hat, b_hat = 1.0, float(np.median(hg[ok] - x[ok])) if ok.sum() >= 20 else 0.0
        else:
            a_hat, b_hat = _fit_affine(x, hg, robust=p.calib_robust)
        lam = float(p.calib_lam)
        a = (1.0 - lam) + lam * a_hat
        b = lam * b_hat
        diag.update(a=a, b=b, a_hat=a_hat, b_hat=b_hat)
    else:
        a, b = 1.0, 0.0
    gr_cal = a * tw_gr + b

    # --- corpus §3 pseudo-typewell: blend the heel's own (TVT_input, GR) in
    if p.pseudo_w > 0 and len(ht) >= 50:
        step = float(np.median(np.diff(tw_tvt))) if len(tw_tvt) > 1 else 0.5
        edges = np.concatenate([tw_tvt - step / 2, [tw_tvt[-1] + step / 2]])
        idx = np.clip(np.searchsorted(edges, ht) - 1, 0, len(tw_tvt) - 1)
        num = np.bincount(idx, weights=hg, minlength=len(tw_tvt))
        cnt = np.bincount(idx, minlength=len(tw_tvt)).astype(float)
        g_heel = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
        k = max(1, int(round(p.pseudo_smooth_ft / max(step, 1e-6))))
        g_heel = _running_mean(g_heel, k)
        mask = _running_mean(np.where(cnt > 0, 1.0, 0.0), k)
        mask = np.nan_to_num(mask, nan=0.0)
        mask = np.where(np.isfinite(g_heel), mask, 0.0)
        g_heel = np.nan_to_num(g_heel, nan=0.0)
        wgt = p.pseudo_w * mask
        gr_ref = (1.0 - wgt) * gr_cal + wgt * g_heel
        diag["pseudo_cover"] = float((mask > 0.5).mean())
    else:
        gr_ref = gr_cal
        diag["pseudo_cover"] = 0.0

    # --- Cauchy scale (U5/U6)
    if p.gamma_mode == "heel_mad" and len(ht) >= 50:
        r = hg - np.interp(ht, tw_tvt, gr_ref)
        g = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        gamma = float(np.clip(g, 1.0, 200.0))
    else:
        gamma = float(p.gamma)
    diag["gamma"] = gamma

    return tw_tvt, gr_ref, gamma, diag


def shuffled_typewell(w, donor):
    """MANDATORY CONTROL.  Return (tvt, gr) with the donor's GR shape placed on
    THIS well's typewell depth axis by relative position.

    Keeping the depth axis is the point: a raw donor curve would simply fall
    outside this well's TVT window and every candidate would score the constant
    out-of-range cost, which tests nothing.  This control keeps depth coverage
    and GR statistics realistic and destroys ONLY the true stratigraphic
    correspondence -- so it isolates the GR channel.
    """
    t, g = np.asarray(w.tw_tvt, float), None
    dt, dg = np.asarray(donor.tw_tvt, float), np.asarray(donor.tw_gr, float)
    if len(t) < 2 or len(dt) < 2:
        return t, np.asarray(w.tw_gr, float)
    u = (t - t[0]) / max(t[-1] - t[0], 1e-9)
    g = np.interp(dt[0] + u * (dt[-1] - dt[0]), dt, dg)
    return t, g


# =============================================================================
# cost tabulation
# =============================================================================

def _segment_costs(C, valid, P, rates, ns, stride_ft, bin_ft, B, oob_cost):
    """seg[l, p, k, b] = likelihood cost of the segment that starts at stage node
    P[p] with start TVT bin b and rate rates[k], running ns[l] samples.

    O(sum(ns_max) * nP * R * B) with the whole (nP,R,B) plane advanced one sample
    at a time -- this is the diagonal accumulation that makes each beam extension
    O(1) instead of O(h_len).
    """
    nP, R, nL = len(P), len(rates), len(ns)
    nmax = int(max(ns))
    maxsh = int(np.ceil(np.abs(rates).max() * (nmax - 1) * stride_ft / bin_ft)) + 2
    W = B + 2 * maxsh
    M = C.shape[0]
    Cpad = np.zeros((M + nmax + 1, W), np.float32)
    Cpad[:M, maxsh:maxsh + B] = C
    # out-of-window bins cost `oob_cost` -- but only on samples that carry GR
    Cpad[:M, :maxsh] = np.where(valid[:, None], oob_cost, 0.0)
    Cpad[:M, maxsh + B:] = np.where(valid[:, None], oob_cost, 0.0)
    # rows past the lateral end stay 0 == the segment is clipped, not penalised (U25)

    bidx = np.arange(B)
    acc = np.zeros((nP, R, B), np.float32)
    seg = np.empty((nL, nP, R, B), np.float32)
    cnt = np.zeros(nP, np.float64)
    cnts = np.empty((nL, nP), np.float64)
    vpad = np.zeros(M + nmax + 1, np.float64)
    vpad[:M] = valid
    li = 0
    order = np.argsort(ns)
    ns_sorted = np.asarray(ns)[order]
    for o in range(nmax):
        sh = np.rint(rates * (o * stride_ft) / bin_ft).astype(np.int64)
        idx = bidx[None, :] + sh[:, None] + maxsh          # (R,B)
        acc += Cpad[P + o][:, idx]                          # (nP,R,B)
        cnt += vpad[P + o]
        while li < nL and ns_sorted[li] == o + 1:
            seg[order[li]] = acc
            cnts[order[li]] = cnt
            li += 1
    while li < nL:                                          # ns longer than the well
        seg[order[li]] = acc
        cnts[order[li]] = cnt
        li += 1
    return seg, cnts


# =============================================================================
# the decode
# =============================================================================

@dataclass
class StrideResult:
    pred_eval: np.ndarray             # TVT on w.is_eval rows, AFTER the chosen read-out
    md_nodes: np.ndarray              # MAP polyline
    tvt_nodes: np.ndarray
    rates: np.ndarray
    cost: float
    diag: dict = field(default_factory=dict)
    # --- read-out material (all on the eval rows) ---------------------------
    map_eval: np.ndarray = None       # the MAP trajectory, always filled
    paths_cost: np.ndarray = None     # (K,) costs of the K cheapest COMPLETE paths, ascending
    paths_eval: np.ndarray = None     # (K, n_eval) their trajectories
    marg_eval: np.ndarray = None      # E[TVT | GR] from the forward-backward, or None
    anchor: float = 0.0               # t0, the hold-last value (shrink target)


def posterior_mean(paths_cost, paths_eval, k=None, tau=1.0):
    """Read-out (1): softmax over COMPLETE paths.  w_j ~ exp(-(c_j - c_min)/tau)."""
    c = np.asarray(paths_cost, float)
    E = np.asarray(paths_eval, float)
    if k:
        c, E = c[:k], E[:k]
    if len(c) == 0:
        return None
    z = -(c - c.min()) / max(float(tau), 1e-9)
    wgt = np.exp(z - z.max())
    wgt /= wgt.sum()
    return (wgt[:, None] * E).sum(0), wgt


def decode_well(w, p: StrideParams | None = None, tw=None) -> StrideResult:
    p = p or StrideParams()
    n, fe = w.n, w.fe
    ev = ~np.isfinite(w.tvt_in)
    n_ev = int(ev.sum())
    if n_ev == 0:
        return StrideResult(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), 0.0, {})

    # ---- anchor: the last known TVT (legal at test time)
    kn = np.where(np.isfinite(w.tvt_in))[0]
    if len(kn) == 0:                                        # no heel at all -> nothing to do
        return StrideResult(np.zeros(n_ev), np.zeros(0), np.zeros(0), np.zeros(0), 0.0,
                            dict(degenerate="no_heel"))
    i0 = int(kn[-1])
    t0 = float(w.tvt_in[i0])
    md0 = float(w.md[i0])
    md_end = float(w.md[-1])
    span = md_end - md0
    if span <= p.stage_ft:
        return StrideResult(np.full(n_ev, t0), np.array([md0, md_end]),
                            np.array([t0, t0]), np.zeros(1), 0.0, dict(degenerate="short"))

    # ---- reference GR curve  (§5 + corpus §3)
    tw_tvt, gr_ref, gamma, diag = _reference_curve(w, p, tw=tw)

    # ---- drilled trend (§6 / U20 reading A): heel dTVT/dMD carried forward
    lo = md0 - p.trend_window_ft
    sel = (w.md[:i0 + 1] >= lo) & np.isfinite(w.tvt_in[:i0 + 1])
    if sel.sum() >= 8:
        xs, ys = w.md[:i0 + 1][sel], w.tvt_in[:i0 + 1][sel]
        trend = _robust_slope(xs, ys) if p.trend_robust else float(np.polyfit(xs, ys, 1)[0])
    else:
        trend = 0.0
    trend_raw = float(np.clip(trend, -p.rate_max, p.rate_max))
    trend = float(np.clip(p.trend_shrink * trend_raw, -p.rate_max, p.rate_max))
    diag["trend_raw"] = trend_raw
    diag["trend_rate"] = trend

    # ---- likelihood samples: block-mean GR along MD (blocks of lik_stride_ft)
    stride_ft = float(p.lik_stride_ft)
    M = int(np.ceil(span / stride_ft)) + 1
    edges = md0 + stride_ft * np.arange(M + 1)
    lat = np.arange(i0, n)
    bi = np.clip(np.searchsorted(edges, w.md[lat]) - 1, 0, M - 1)
    use = w.gr_obs[lat] & ev[lat]                            # never score the heel; never score filled GR
    num = np.bincount(bi[use], weights=w.gr[lat][use], minlength=M)
    cnt = np.bincount(bi[use], minlength=M).astype(float)
    gr_s = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
    valid = np.isfinite(gr_s)

    # ---- TVT bins
    bin_ft = float(p.tvt_bin)
    b0 = int(round(p.tvt_halfrange / bin_ft))
    B = 2 * b0 + 1
    tvt_bins = t0 + (np.arange(B) - b0) * bin_ft
    gref_b = np.interp(tvt_bins, tw_tvt, gr_ref)
    inrange = (tvt_bins >= tw_tvt[0]) & (tvt_bins <= tw_tvt[-1])

    # ---- effective likelihood weight (global temperature, optionally length-normalised)
    lw = float(p.lik_weight)
    if p.lik_norm_ref > 0:
        lw = lw * float(p.lik_norm_ref) / max(int(valid.sum()), 1)
    diag["lik_weight_eff"] = lw

    # ---- per-sample cost table C[m, b]
    r = gr_s[:, None] - gref_b[None, :]
    if p.misfit == "cauchy":
        Cm = np.log1p((r / gamma) ** 2)
    else:
        Cm = 0.5 * (r / gamma) ** 2
    Cm = np.where(inrange[None, :], Cm, p.oob_cost)
    Cm = (lw * np.where(valid[:, None], Cm, 0.0)).astype(np.float32)

    # ---- lattice
    rates = np.arange(-p.rate_max, p.rate_max + 1e-12, p.rate_step)
    if len(rates) % 2 == 0:
        rates = np.append(rates, rates[-1] + p.rate_step)
    R = len(rates)
    Lft = np.array([l for l in p.hlen_ft], float)
    Lstage = np.maximum(1, np.rint(Lft / p.stage_ft).astype(int))
    Lft = Lstage * p.stage_ft                               # snap to the stage grid
    nL = len(Lft)
    ns = np.maximum(1, np.rint(Lft / stride_ft).astype(int))

    S = int(np.ceil(span / p.stage_ft))                     # terminal stage index
    P = np.rint((p.stage_ft / stride_ft) * np.arange(S)).astype(int)
    P = np.clip(P, 0, M - 1)

    seg, segcnt = _segment_costs(Cm, valid.astype(float), P, rates, ns,
                                 stride_ft, bin_ft, B, float(lw * p.oob_cost))
    if p.tau > 0:
        seg = seg / np.maximum(segcnt, 1.0)[:, :, None, None] ** p.tau

    # ---- priors
    cost_len = (np.log(Lft) - p.len_mu) ** 2 / (2 * p.len_sigma ** 2)
    if p.len_jacobian:
        cost_len = cost_len + np.log(Lft)

    def _pers(dr, L, sig):
        z2 = (dr / sig) ** 2
        scale = L ** (2.0 - p.pers_p)
        if p.pers_kernel == "gauss":
            return 0.5 * scale * z2
        return np.log1p(scale * z2)

    # pers[k_prev, l, k_new]
    centre = (1.0 - p.mean_revert) * rates[:, None] + p.mean_revert * trend      # (R,1)
    dr = rates[None, :] - centre                                                  # (R_prev, R_new)
    pers = _pers(dr[:, None, :], Lft[None, :, None], p.sigma_p).astype(np.float32)
    pers0 = _pers(rates - trend, Lft[:, None], p.sigma_p * p.init_sigma_mult).astype(np.float32)  # (nL,R)

    # ---- beam / DP
    INF = np.float32(1e18)
    tau_post = max(float(p.post_tau), 1e-9)
    want_alpha = (p.readout == "marginal")
    st_cost = [None] * (S + 1)
    st_tvt = [None] * (S + 1)
    st_bin = [None] * (S + 1)
    st_k = [None] * (S + 1)
    st_bps = [None] * (S + 1)
    st_bpi = [None] * (S + 1)
    st_l = [None] * (S + 1)
    st_ac = [None] * (S + 1)          # forward LOG-SUM-EXP score (tempered), for "marginal"
    pend = [[] for _ in range(S + 1)]

    term = []   # (cost, tvt, from_stage, from_idx, l, k, acost)

    # seed: stage 0, exactly one state (bin b0, "rate" slot irrelevant -> use pers0)
    cur = dict(cost=np.zeros(1, np.float32), tvt=np.array([t0]),
               bin=np.array([b0], np.int64), k=np.array([-1], np.int64),
               bps=np.array([-1], np.int64), bpi=np.array([-1], np.int64),
               l=np.array([-1], np.int64), ac=np.zeros(1, np.float64))

    for pst in range(S):
        if pst > 0:
            cand = pend[pst]
            if not cand:
                st_cost[pst] = None
                continue
            cost = np.concatenate([c[0] for c in cand])
            tvt = np.concatenate([c[1] for c in cand])
            bn = np.concatenate([c[2] for c in cand])
            kk = np.concatenate([c[3] for c in cand])
            bs = np.concatenate([c[4] for c in cand])
            bp = np.concatenate([c[5] for c in cand])
            ll = np.concatenate([c[6] for c in cand])
            ac = np.concatenate([c[7] for c in cand])
            pend[pst] = []
            keep = min(len(cost), p.beam * p.beam_overfetch)
            if len(cost) > keep:
                sel = np.argpartition(cost, keep - 1)[:keep]
                cost, tvt, bn, kk, bs, bp, ll, ac = (cost[sel], tvt[sel], bn[sel], kk[sel],
                                                     bs[sel], bp[sel], ll[sel], ac[sel])
            order = np.argsort(cost, kind="stable")
            cost, tvt, bn, kk, bs, bp, ll, ac = (cost[order], tvt[order], bn[order], kk[order],
                                                 bs[order], bp[order], ll[order], ac[order])
            if p.dedup:
                key = bn * R + kk
                # cost is ascending, so np.unique's first occurrence per key IS the Viterbi
                # representative.  For the sum-product score the duplicates are not dropped,
                # they are LOG-SUM-EXP'd into the survivor -- exact, because the state really
                # is Markov in (tvt_bin, rate) (D3).
                uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
                inv = inv.reshape(-1)
                agrp = None
                if want_alpha and len(uk) < len(key):
                    amin = np.full(len(uk), np.inf)
                    np.minimum.at(amin, inv, ac)
                    ssum = np.bincount(inv, weights=np.exp(-(ac - amin[inv])), minlength=len(uk))
                    agrp = amin - np.log(np.maximum(ssum, 1e-300))
                o = np.argsort(first)
                first = first[o]
                cost, tvt, bn, kk, bs, bp, ll, ac = (cost[first], tvt[first], bn[first], kk[first],
                                                     bs[first], bp[first], ll[first], ac[first])
                if agrp is not None:
                    ac = agrp[o]
            cost, tvt, bn, kk, bs, bp, ll, ac = (cost[:p.beam], tvt[:p.beam], bn[:p.beam],
                                                 kk[:p.beam], bs[:p.beam], bp[:p.beam],
                                                 ll[:p.beam], ac[:p.beam])
            cur = dict(cost=cost, tvt=tvt, bin=bn, k=kk, bps=bs, bpi=bp, l=ll, ac=ac)

        st_cost[pst] = cur["cost"]; st_tvt[pst] = cur["tvt"]; st_bin[pst] = cur["bin"]
        st_k[pst] = cur["k"]; st_bps[pst] = cur["bps"]; st_bpi[pst] = cur["bpi"]
        st_l[pst] = cur["l"]; st_ac[pst] = cur["ac"]

        m = len(cur["cost"])
        if m == 0:
            continue

        # likelihood of every (l, k) extension from every live state -> (m, nL, R)
        lik = np.transpose(seg[:, pst][:, :, cur["bin"]], (2, 0, 1))
        if pst == 0:
            pr = np.broadcast_to(pers0[None, :, :], (m, nL, R))
        else:
            pr = pers[cur["k"]]
        edge = lik + pr + cost_len[None, :, None].astype(np.float32)
        tot = cur["cost"][:, None, None] + edge

        tvt_new = cur["tvt"][:, None, None] + rates[None, None, :] * Lft[None, :, None]
        bin_new = np.rint((tvt_new - t0) / bin_ft).astype(np.int64) + b0
        okb = (bin_new >= 0) & (bin_new < B)

        mi, li_, ki = np.nonzero(okb)
        if len(mi) == 0:
            continue
        c_ = tot[mi, li_, ki]
        t_ = tvt_new[mi, li_, ki]
        b_ = bin_new[mi, li_, ki]
        a_ = (cur["ac"][mi] + edge[mi, li_, ki].astype(np.float64) / tau_post
              if want_alpha else np.zeros(len(mi)))
        tgt = pst + Lstage[li_]

        done = tgt >= S
        if done.any():
            d = np.nonzero(done)[0]
            j = np.argsort(c_[d])[:p.beam]
            d = d[j]
            term.append((c_[d], t_[d], np.full(len(d), pst), mi[d], li_[d], ki[d], a_[d]))

        alive = ~done
        if alive.any():
            ai = np.nonzero(alive)[0]
            for s in np.unique(tgt[ai]):
                q = ai[tgt[ai] == s]
                pend[s].append((c_[q], t_[q], b_[q], ki[q],
                                np.full(len(q), pst, np.int64), mi[q], li_[q], a_[q]))

    # ---- terminal pick + backtrack
    if not term:
        pred = np.full(n_ev, t0)
        return StrideResult(pred, np.array([md0, md_end]), np.array([t0, t0]),
                            np.zeros(1), float("inf"), dict(degenerate="no_terminal", **diag))

    tc = np.concatenate([t[0] for t in term])
    tt = np.concatenate([t[1] for t in term])
    tp = np.concatenate([t[2] for t in term])
    tm = np.concatenate([t[3] for t in term])
    tl = np.concatenate([t[4] for t in term])
    tk = np.concatenate([t[5] for t in term])
    ta = np.concatenate([t[6] for t in term])
    ordt = np.argsort(tc, kind="stable")
    j = int(ordt[0])
    best_cost = float(tc[j])
    md_ev = w.md[ev]

    def _polyline(jj):
        """backtrack terminal candidate jj -> (md_nodes, tvt_nodes, rates)."""
        segs = [(int(tp[jj]), int(tl[jj]), int(tk[jj]))]     # (start_stage, l_idx, k_idx)
        st, ix = int(tp[jj]), int(tm[jj])
        while st > 0:
            segs.append((int(st_bps[st][ix]), int(st_l[st][ix]), int(st_k[st][ix])))
            st, ix = int(st_bps[st][ix]), int(st_bpi[st][ix])
        segs.reverse()
        mdn = [md0]; tvn = [t0]; rr = []
        cm, ct = md0, t0
        for (_s, li_, ki) in segs:
            Lb = float(Lft[li_])
            ct = ct + rates[ki] * Lb
            cm = cm + Lb
            mdn.append(cm); tvn.append(ct); rr.append(float(rates[ki]))
        return np.array(mdn), np.array(tvn), np.array(rr)

    md_nodes, tvt_nodes, rs = _polyline(j)
    map_eval = np.interp(md_ev, md_nodes, tvt_nodes)         # linear inside a segment (U1)

    # ---- READ-OUT (1): the K cheapest COMPLETE paths, for the softmax mean ----
    # DIVERSITY IS NOT FREE.  From one surviving pre-terminal state there are nL*R ways to
    # finish, and they all cost about the same, so the K cheapest terminals are K variations
    # of the LAST SEGMENT of a single path (measured: mean sd across the top 64 = 0.38 ft).
    # `post_dedup_parent` keeps only the cheapest finish per pre-terminal state, so the K
    # paths differ in their whole history, which is what a posterior mean is supposed to
    # average over.
    K = int(max(p.post_k if p.readout == "beam_mean" else 0, p.keep_paths, 0))
    paths_cost = paths_eval = None
    if K > 0:
        if p.post_dedup_parent:
            pkey = tp[ordt] * (int(tm.max()) + 2) + tm[ordt]
            _, fst = np.unique(pkey, return_index=True)
            ordt = ordt[np.sort(fst)]
        selt = ordt[:K]
        pe = np.empty((len(selt), n_ev))
        for q, jj in enumerate(selt):
            mdn, tvn, _ = _polyline(int(jj))
            pe[q] = np.interp(md_ev, mdn, tvn)
        paths_cost = tc[selt].astype(float)
        paths_eval = pe
        diag["path_spread"] = float(np.mean(np.std(pe, axis=0)))     # ft, mean over eval rows
        diag["path_dcost"] = float(paths_cost[-1] - paths_cost[0])

    # ---- READ-OUT (2): proper marginal posterior mean, forward-backward ------
    marg_eval = None
    if want_alpha:
        marg_eval = _marginal_mean(
            S, nL, R, B, b0, bin_ft, t0, md0, p.stage_ft, Lft, Lstage, rates, seg,
            pers, pers0, cost_len, st_bin, st_k, st_tvt, st_ac, tau_post, md_ev, diag,
            float(p.marg_min_mass))

    # ---- pick, then the shrink-to-anchor CONTROL ----------------------------
    if p.readout == "beam_mean" and paths_cost is not None:
        pred_all = posterior_mean(paths_cost, paths_eval, p.post_k, p.post_tau)[0]
    elif p.readout == "marginal" and marg_eval is not None:
        pred_all = marg_eval
    else:
        pred_all = map_eval
    if p.shrink_w > 0:
        pred_all = (1.0 - p.shrink_w) * pred_all + p.shrink_w * t0

    diag.update(n_seg=len(rs), stages=S, beam_used=p.beam,
                lattice=int(nL * R), samples=int(valid.sum()), n_term=int(len(tc)))
    return StrideResult(pred_all, md_nodes, tvt_nodes, np.array(rs), best_cost, diag,
                        map_eval=map_eval, paths_cost=paths_cost, paths_eval=paths_eval,
                        marg_eval=marg_eval, anchor=t0)


# =============================================================================
# read-out (2): the marginal posterior mean  E[TVT(md) | GR]
# =============================================================================

def _marginal_mean(S, nL, R, B, b0, bin_ft, t0, md0, stage_ft, Lft, Lstage, rates, seg,
                   pers, pers0, cost_len, st_bin, st_k, st_tvt, st_ac, tau, md_ev, diag, min_mass):
    """Sum-product over the retained trellis, then E[TVT] assembled from SEGMENT marginals.

    Why a segment (edge) marginal and not a node marginal: a path only has a node at the
    stages where a segment starts, so node marginals do not cover the MD axis.  Every edge,
    however, covers a contiguous run of stage nodes with a TVT that is LINEAR in the node
    index, so E[TVT(node j)] = sum_e p(e) * (t_e + r_e*(j-s_e)*stage_ft) is exactly assembled
    with two difference arrays, and E[TVT(md)] is then exactly piecewise linear between the
    stage nodes (all segment boundaries live on that grid).  Cost: two extra sweeps.
    """
    cl = cost_len[None, :, None].astype(np.float32)
    # ---------- backward: beta(state) = -log sum_suffixes exp(-cost/tau) ----------
    beta = [None] * (S + 1)
    tab = np.full(B * R, -1, np.int64)
    kgrid = np.arange(R)[None, :]
    for s in range(S - 1, -1, -1):
        bins_s = st_bin[s]
        if bins_s is None or len(bins_s) == 0:
            continue
        ms = len(bins_s)
        lik = np.transpose(seg[:, s][:, :, bins_s], (2, 0, 1))
        pr = np.broadcast_to(pers0[None, :, :], (ms, nL, R)) if s == 0 else pers[st_k[s]]
        edge = (lik + pr + cl).astype(np.float64) / tau
        tvt_new = st_tvt[s][:, None, None] + rates[None, None, :] * Lft[None, :, None]
        bin_new = np.rint((tvt_new - t0) / bin_ft).astype(np.int64) + b0
        okb = (bin_new >= 0) & (bin_new < B)
        btot = np.full((ms, nL, R), np.inf)
        for li in range(nL):
            tgt = s + Lstage[li]
            if tgt >= S:                                   # clipped final segment: beta = 0
                mk = okb[:, li, :]
                bl = btot[:, li, :]; bl[mk] = edge[:, li, :][mk]; btot[:, li, :] = bl
                continue
            bnN, kkN, beN = st_bin[tgt], st_k[tgt], beta[tgt]
            if bnN is None or len(bnN) == 0 or beN is None:
                continue
            tab[bnN * R + kkN] = np.arange(len(bnN))
            idx = tab[np.clip(bin_new[:, li, :], 0, B - 1) * R + kgrid]
            good = okb[:, li, :] & (idx >= 0)
            bl = np.full((ms, R), np.inf)
            bl[good] = beN[idx[good]]
            btot[:, li, :] = edge[:, li, :] + bl
            tab[bnN * R + kkN] = -1
        bmin = btot.min(axis=(1, 2))
        fin = np.isfinite(bmin)
        sh = np.where(fin, bmin, 0.0)[:, None, None]
        ssum = np.exp(-(btot - sh)).sum(axis=(1, 2))
        beta[s] = np.where(fin & (ssum > 0), bmin - np.log(np.maximum(ssum, 1e-300)), np.inf)
    if beta[0] is None or not np.isfinite(beta[0][0]):
        return None
    F = float(beta[0][0])                                  # = -log Z  (alpha at the seed is 0)

    # ---------- forward sweep: edge posteriors -> E[TVT] on the stage nodes ----------
    D0 = np.zeros(S + 3); D1 = np.zeros(S + 3); DM = np.zeros(S + 3)
    for s in range(S):
        bins_s = st_bin[s]
        if bins_s is None or len(bins_s) == 0:
            continue
        ms = len(bins_s)
        lik = np.transpose(seg[:, s][:, :, bins_s], (2, 0, 1))
        pr = np.broadcast_to(pers0[None, :, :], (ms, nL, R)) if s == 0 else pers[st_k[s]]
        edge = (lik + pr + cl).astype(np.float64) / tau
        tvt_new = st_tvt[s][:, None, None] + rates[None, None, :] * Lft[None, :, None]
        bin_new = np.rint((tvt_new - t0) / bin_ft).astype(np.int64) + b0
        okb = (bin_new >= 0) & (bin_new < B)
        tot = np.full((ms, nL, R), np.inf)
        for li in range(nL):
            tgt = s + Lstage[li]
            if tgt >= S:
                mk = okb[:, li, :]
                bl = tot[:, li, :]; bl[mk] = edge[:, li, :][mk]; tot[:, li, :] = bl
                continue
            bnN, kkN, beN = st_bin[tgt], st_k[tgt], beta[tgt]
            if bnN is None or len(bnN) == 0 or beN is None:
                continue
            tab[bnN * R + kkN] = np.arange(len(bnN))
            idx = tab[np.clip(bin_new[:, li, :], 0, B - 1) * R + kgrid]
            good = okb[:, li, :] & (idx >= 0)
            bl = np.full((ms, R), np.inf)
            bl[good] = beN[idx[good]]
            tot[:, li, :] = edge[:, li, :] + bl
            tab[bnN * R + kkN] = -1
        tot = tot + st_ac[s][:, None, None]
        keep = tot < (F + 25.0)                            # drop edges below exp(-25) of the mass
        if not keep.any():
            continue
        ii, lj, kj = np.nonzero(keep)
        pw = np.exp(F - tot[ii, lj, kj])
        tj = st_tvt[s][ii]
        rj = rates[kj]
        end = np.where(s + Lstage[lj] >= S, S + 1, s + Lstage[lj])
        A = pw * ((tj - t0) - rj * stage_ft * s)          # anchored at t0: conditioning
        Bc = pw * rj * stage_ft
        st_i = np.full(len(ii), s)
        n_ = S + 3
        D0 += np.bincount(st_i, weights=A, minlength=n_) - np.bincount(end, weights=A, minlength=n_)
        D1 += np.bincount(st_i, weights=Bc, minlength=n_) - np.bincount(end, weights=Bc, minlength=n_)
        DM += np.bincount(st_i, weights=pw, minlength=n_) - np.bincount(end, weights=pw, minlength=n_)
    C0 = np.cumsum(D0)[:S + 1]
    C1 = np.cumsum(D1)[:S + 1]
    CM = np.cumsum(DM)[:S + 1]
    diag["marg_mass_min"] = float(CM.min())
    diag["marg_mass_max"] = float(CM.max())
    if CM.min() < min_mass:
        return None                                   # truncation, not a posterior -- refuse
    node_md = md0 + stage_ft * np.arange(S + 1)
    node_tvt = t0 + (C0 + C1 * np.arange(S + 1)) / CM
    return np.interp(md_ev, node_md, node_tvt)


def predict_well(w, params: StrideParams | None = None, tw=None) -> np.ndarray:
    """Predicted TVT on the eval rows.  len == (~isfinite(w.tvt_in)).sum()."""
    return decode_well(w, params, tw=tw).pred_eval
