"""rogii.coefnet -- ONE WELL = ONE SAMPLE, and the OUTPUT IS NINE NUMBERS.

THE ORGAN THAT WAS MISSING.  Every sequence model in this repo so far emits a
per-STEP quantity (seqwell: dTVT ; train_seq: an offset per row) and is then
integrated.  docs/README Sec.2 records the consequence three times over, on three
independent routes: **the metric is on the INTEGRAL, so a prior that is right
locally and biased globally is worse than no prior at all** (seqwell's -dZ leaves
0.022 ft/step x 4 700 steps = 90-230 ft of drift; STRIDE picks 15.89 against a
family oracle of 9.18; the spatial model reads CV 10.72 and LB 15.34).

This module cannot commit that error, by construction: **it never produces a
per-step prediction at all.**  It reads the whole well and emits the INTEGRAL
directly -- the 9 Legendre coefficients of the champion's residual.

WHY THIS SHAPE, AND WHY IT HAS NEVER BEEN TRIED
-----------------------------------------------
  * docs/25 + docs/26: the residual of `v5_nospatial_clean` is a SMOOTH LOW
    DEGREE CURVE in normalised MD.  deg-8 = 98.42 % of the SSE, ~10.3 effective
    DOF per well, ACF L(1/e) = 462 ft, and the 15-25 ft bedding period is 134x
    DEFICIENT against white noise.  The problem is "emit ~9 numbers per well",
    not "predict 1 200 rows".
  * docs/29 (FRONT E) regressed exactly these 9 numbers and FAILED: only `a0`
    is predictable (R2 OOF +0.0401, 18 sigma), `a1..a8` are all at or below
    their permutation null.
  * docs/36 Sec.2.2 found the hole in that verdict: `coeff_regress.py:133` --
    the 738 features are, WITHOUT EXCEPTION, SCALARS PER WELL.  The regressor
    sees ONE ROW PER WELL.  The correct statement is "the coefficients are not
    predictable FROM SUMMARIES".  The path SEQUENCE -> COEFFICIENTS has never
    been tested.  And the same experiment measures that it is the AGGREGATION
    that destroys the signal (the 73 aggregated `w_*` descriptors are worth
    R2 -0.0038; the 680 along-the-toe statistics carry everything).
  * docs/38 Sec.4 (Q5) closed the remaining escape: the target is NOT model
    noise.  Two LGBMs with different feature sets make the SAME error on each
    well, corr 0.90 -> 0.98 on all nine coefficients, against a permutation
    null of 0.03.  `a_j` is a property of the WELL.

THE LOSS IS THE METRIC -- exactly, with the audit's correction
--------------------------------------------------------------
`docs/38` Q1 measured that the naive thesis ("SSE = n * sum_j (a_j-ahat_j)^2")
is FALSE by an additive term, and gave the exact relation:

    RMSE_pooled(ahat)^2 = [ sum_w ||r_w||^2 + sum_w n_w * sum_j (a_wj-ahat_wj)^2 ]
                          / sum_w n_w
                          \___ CONSTANT ___/  \_____ the loss, WEIGHTED by n_w ____/

Consequences, all of them wired into this file:
  1. `||r_w||^2` (the energy outside the deg-8 span, 1.5806 % of the SSE) does
     NOT depend on ahat  =>  same gradient, same argmin.  Training on the
     coefficient MSE optimises the pooled RMSE.  `coef_loss` is that MSE.
  2. The loss is NEVER a number of feet.  It is offset by -1.6765^2 and
     rescaled, and the gap between the two readings reaches 4.4327 ft on one
     well.  **Every gate in this file is written in RECONSTRUCTED RMSE**
     (`reconstructed_rmse`), never in training loss.
  3. The per-well weight is `n_w` = its number of EVALUATED rows (docs/38 Q2:
     n varies 24.64x, 102 -> 2 513).  It is one scalar per sample, it is free,
     and it is worth ~0.01 ft -- apply it, expect nothing from it.
  4. The basis `P` is the well's OWN (QR of the Legendre Vandermonde on the
     `tn` of its evaluated span).  A frozen global basis BREAKS the identity.
     `eval_grid` rebuilds it from `w.md` / `w.fe` alone, which is test-safe.

THE METRIC'S OWN WEIGHTING IS THE RIGHT WEIGHTING.  `a0` has sd 10.776 ft and
carries 67.82 % of the SSE; `a8` has sd 0.715 ft and carries 0.32 %.  The head
is therefore NOT output-normalised: in feet-squared the gradient on `a0` is
~225x the gradient on `a8`, and that ratio IS the exchange rate the leaderboard
pays.  Normalising the outputs would silently buy `a8` (worth -0.0212 ft
perfect, docs/38 Table 2) at the price of `a0` (worth -5.7708 ft perfect).

ZERO-INIT = THE BASE, EXACTLY.  The last linear layer is zero-initialised, so an
untrained net emits ahat = 0, i.e. "do nothing", i.e. RMSE = 13.335168890045058
to the bit.  The model starts ON the gate it has to beat.  (Same device as
seqwell's U6, for the same reason: under a short budget, where the zero-init
model sits is most of the outcome.)

BIDIRECTIONAL IS LEGAL HERE, and it is the point.  seqwell's decoder is
autoregressive because it emits a step at a time and each step feeds the next.
Nothing is causal here: the well is read IN FULL before a single number is
emitted, exactly as Tucker Arrants describes the attested sub-5 object
("not a tabular model, so each sample is a complete well", docs/README Sec.6).

TEST SAFETY.  Every input is the well's own MD / X / Y / Z / GR / gr_obs /
TVT_input / fe and its own typewell TVT/GR.  `w.tvt`, `w.tops`, `w.tw_geol`,
`w.datum` are NEVER read on any path in this file -- the truth enters ONLY as
the training target `a`, which is loaded from `data/derived/coeff_targets_n773.npz`
(docs/38 Q3: out-of-fold, reproduced bit-for-bit, legitimate) and is consumed by
the loss and the scorer alone.  `truth_independence_check` proves it at runtime
by destroying those four fields and asserting the input tensor does not move by
one bit.

STRICTLY INTRA-WELL.  No neighbour well, no spatial field, no surface
coordinate.  The extra per-row channels are the ones `rogii/seqwell.py` already
audits (`audit_channels` refuses every `CROSS_WELL_PREFIXES` match), and this
file calls that same auditor rather than re-implementing the list.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from rogii.seqwell import N_BASE, FEAT_SCALE           # noqa: F401  (7 base channels)

# --------------------------------------------------------------------------- #
# the pre-registered gates.  docs/29 Sec.0.3 caught a model beating its own
# null by -0.49 ft while being +0.19 ft WORSE than doing nothing, hence TWO
# gates, and hence BASE (not the null) is the reference.
# --------------------------------------------------------------------------- #
#: runs/v5_nospatial_clean, pooled OOF RMSE, bit-exact (meta.json)
BASE_RMSE = 13.335168890045058
#: docs/29 Sec.5.1 -- ridge on 656 per-well summaries, `a0` ALONE (R2 +0.0401).
#: the floor a summary regressor already reaches; below this we are not useful.
RIDGE_C0_RMSE = 13.1484
#: docs/38 Table 1 -- `a` PERFECT.  The energy outside the deg-8 span.
DEG8_FLOOR_RMSE = 1.6765

DMAX = 8                 #: degree of the Legendre truncation -> 9 coefficients
NCOEF = DMAX + 1
EVAL_SUB = 4             #: feat_v6 / submission row grid = np.arange(fe, n)[::4]

#: MEASURED ceiling on |channel| (p99.9 over the loaded corpus).  The `x` block
#: carries FIXED pre-scales and no fitted statistic, so a channel whose units are
#: wrong is silent: it just eats the projection.  Measured 2026-07-26 on 40
#: fold-balanced wells / 275 929 rows: every channel sits at p99.9 <= 2.5 EXCEPT
#: `dz` (20.0, it is FEAT_SCALE'd), `dtvt_in` (35.2) and `heel_off` (**92.45**).
#: At initialisation `heel_off` alone owns **68.7 %** of Var(proj(x)) and
#: `grn` -- the GR log, the physics -- owns **0.2 %**.  Post-LayerNorm the
#: damage is bounded (||dh||/||h|| 0.3066 for heel_off vs 0.2564 for grn), which
#: is why this is a TRIPWIRE at 100 and not a silent rescale: the ceiling is 8 %
#: away from the worst live channel, so ANY new channel in raw feet trips it.
MAX_CHANNEL_AMPLITUDE = 100.0


# =========================================================================== #
# 1.  THE WELL'S OWN BASIS -- test-safe, rebuilt from md/fe/n alone            #
# =========================================================================== #
def ortho_basis(tn: np.ndarray, dmax: int = DMAX) -> np.ndarray:
    """Q (n x dmax+1) orthonormal, signs FIXED by diag(R) > 0.

    BIT-IDENTICAL to `tools/coeff_regress.ortho_basis` -- it must be, because the
    target `a` in the cache was produced by that function and a different sign
    convention would silently flip a coefficient on some wells.  The smoke
    asserts the two agree to 0 on real wells rather than trusting this comment.
    """
    V = np.polynomial.legendre.legvander(tn, dmax)
    Q, R = np.linalg.qr(V)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return Q * s


def eval_grid(w, sub: int = EVAL_SUB):
    """(rows, tn) for the EVALUATED span of well `w`.

    Reads `w.md`, `w.fe`, `w.n` and NOTHING else -- available at test time.
    `rows` reproduces `rogii/features.build_well_features` (`np.arange(fe,n)[::sub]`)
    and `tn` reproduces `tools/coeff_regress.build_cache` line by line
    (`dmd = md[idx] - md[i0]` with `i0 = fe-1`, then `tn = 2(d-d0)/L - 1`).
    Verified against the cache's own `n_rows` on all 773 wells by the smoke.
    """
    n, fe = int(w.n), int(w.fe)
    rows = np.arange(fe, n)[::sub]
    md = np.asarray(w.md, float)
    d = md[rows] - md[fe - 1]
    L = d[-1] - d[0]
    tn = 2.0 * (d - d[0]) / (L if L > 0 else 1.0) - 1.0
    return rows, tn


# =========================================================================== #
# 2.  PER-ROW POSITION CHANNELS -- what an encoder needs and a decoder did not #
# =========================================================================== #
#: seqwell is autoregressive and step-local, so it never needed to know WHERE in
#: the well a row sits.  An encoder that emits a coefficient of `tn` does: a
#: pooling that cannot tell -1 from +1 cannot emit an ODD coefficient at all.
#:
#: `known` WAS the third member and is GONE (2026-07-26).  It was documented as
#: "not simply 1 - is_eval: docs/33 measured wells with interior known rows",
#: and that claim is FALSE **in this loader**: `dataset.py:60-61` sets
#: `fe = argmax(~isfinite(tvt_in))`, i.e. the FIRST NaN, so a known row past `fe`
#: cannot exist by construction.  Measured: `max |known - (1 - is_eval)| =
#: 0.000e+00` over 40 fold-balanced wells / 275 929 rows here, and 773/773 wells
#: in the refuter's pass.  It was an EXACT affine copy of `is_eval`, i.e. a
#: 12th channel carrying zero bits.  `duplicate_channels()` now refuses it, so it
#: cannot come back in silence.
POS_CHANNELS = ("tn", "is_eval", "heel_off", "dtvt_in")

#: the 7 seqwell base channels, in `WellSeq.base` order (rogii/seqwell.py:965-973).
BASE_CHANNEL_NAMES = ("dmd", "dx", "dy", "dz", "grn", "dgrn", "gr_obs")

#: the four `Well` fields that ARE the answer.  No input channel may declare one.
TRUTH_FIELDS = ("tvt", "tops", "tw_geol", "datum")

#: WHICH `Well` FIELDS EACH STACKED CHANNEL READS.  This is the object the leak
#: audit needs and did not have: `audit_channels` (rogii/seqwell.py:674) only
#: ever saw the `extra` block, which is EMPTY by default -- so it printed eight
#: reassuring lines about `violations = []` while the 11 channels that actually
#: reach the encoder were never shown to it.  Every entry here is checked against
#: TRUTH_FIELDS *and* against seqwell's CROSS_WELL_PREFIXES, and the total count
#: is checked against the width of the tensor that is really packed.
CHANNEL_PROVENANCE = {
    "dmd": ("md",),                 "dx": ("x",),
    "dy": ("y",),                   "dz": ("z",),
    "grn": ("gr", "tw_gr"),         "dgrn": ("gr", "tw_gr"),
    "gr_obs": ("gr_obs",),
    "tn": ("md", "fe", "n"),        "is_eval": ("fe", "n"),
    "heel_off": ("tvt_in", "fe"),   "dtvt_in": ("tvt_in",),
}


def pos_channel_block(w, names=POS_CHANNELS, sub: int = EVAL_SUB) -> np.ndarray:
    """The position / anchor channels, per row, on `np.arange(w.n)`.

    Reads `md, fe, n, tvt_in` -- all given at test time.  `w.tvt` is NOT read.

      tn        the SAME normalised MD the target's basis uses, interpolated onto
                every row and edge-clamped on the heel (exact at the eval rows).
      is_eval   1 past the PS, 0 on the known lateral.
      heel_off  (TVT_input - anchor)/10 where known, 0 elsewhere.  The anchor is
                `tvt_in[fe-1]`, the same one `rogii/features.py:128` uses.
                MEASURED HEEL-ONLY: sd 23.15 on the heel, **0.0000 on the whole
                evaluated span** (100 % exact zeros) -- TVT_input does not exist
                past the PS, so this channel CANNOT be alive there.  It is kept
                because it is one of the only two windows on TVT_input at all,
                and `channel_census` prints that asymmetry every run instead of
                letting the docstring imply the model sees it on the toe.
      dtvt_in   the per-step drift of the KNOWN track x20, 0 past the PS.  Same
                heel-only status (sd 8.62 / 0.0000).  It is the exact finite
                difference of `heel_off` (`max |dtvt_in - 200*diff(heel_off)| =
                1.5e-03`, float32 rounding) -- redundant to a conv the way
                seqwell's own `dgrn` is redundant to `grn`, and kept for the same
                reason: the first op after `proj` is per-row, not a difference.
    """
    n, fe = int(w.n), int(w.fe)
    ti = np.asarray(w.tvt_in, float)
    known = np.isfinite(ti)
    anchor = float(ti[fe - 1]) if fe > 0 else float(np.nanmedian(ti))
    rows, tn_e = eval_grid(w, sub)
    out = {}
    out["tn"] = np.interp(np.arange(n, dtype=float), rows.astype(float), tn_e)
    out["is_eval"] = (np.arange(n) >= fe).astype(float)
    ho = np.where(known, np.nan_to_num(ti - anchor), 0.0) * 0.1
    out["heel_off"] = ho
    dt = np.zeros(n)
    dt[1:] = np.where(known[1:] & known[:-1],
                      np.nan_to_num(np.diff(np.nan_to_num(ti))), 0.0)
    out["dtvt_in"] = dt * 20.0
    miss = [c for c in names if c not in out]
    if miss:
        raise KeyError("unknown position channel(s): %s (available: %s)"
                       % (miss, sorted(out)))
    return np.stack([out[c] for c in names], 1).astype(np.float32)


# --------------------------------------------------------------------------- #
# 2b.  THE LEAK AUDIT THAT SEES THE CHANNELS THAT ARE REALLY STACKED           #
# --------------------------------------------------------------------------- #
def stacked_channel_names(pos_names=POS_CHANNELS, extra_names=()) -> list:
    """base(7) | position | extra -- the EXACT order `CoefSample.x` concatenates."""
    return list(BASE_CHANNEL_NAMES) + list(pos_names) + list(extra_names)


def audit_stacked_channels(names, n_chan: int, extra_names=(), verbose=True) -> dict:
    """LEAK AUDIT over EVERY channel the encoder receives.  Raises on violation.

    Five ways this fails, all reachable:
      1. a stacked channel with no CHANNEL_PROVENANCE entry (someone added a
         channel to POS_CHANNELS and forgot to declare where it comes from);
      2. a declared provenance that names one of TRUTH_FIELDS;
      3. a channel name matching one of seqwell's 25 CROSS_WELL_PREFIXES --
         now applied to the base and position blocks too, not only to `extra`;
      4. `len(names) != n_chan`, i.e. the audited list and the packed tensor have
         drifted apart -- the failure mode that made the old audit vacuous;
      5. a duplicate channel name.
    """
    from rogii.seqwell import CROSS_WELL_PREFIXES

    rep = {"names": list(names), "n_chan": int(n_chan), "violations": []}
    extra = set(extra_names)
    for c in names:
        if c in extra:
            continue                        # audited by seqwell.audit_channels
        prov = CHANNEL_PROVENANCE.get(c)
        if prov is None:
            rep["violations"].append(
                "%s: STACKED but UNDECLARED (no CHANNEL_PROVENANCE entry)" % c)
            continue
        bad = [f for f in prov if f in TRUTH_FIELDS]
        if bad:
            rep["violations"].append("%s: reads TRUTH field(s) %s" % (c, bad))
    for c in names:
        for pre in CROSS_WELL_PREFIXES:
            if c.startswith(pre):
                rep["violations"].append("%s: cross-well prefix %r" % (c, pre))
    if len(names) != int(n_chan):
        rep["violations"].append(
            "audited %d channel names but the packed tensor has %d channels: "
            "the audit and the model have drifted apart"
            % (len(names), int(n_chan)))
    dup = sorted({c for c in names if list(names).count(c) > 1})
    if dup:
        rep["violations"].append("duplicate channel name(s): %s" % dup)
    if verbose:
        print("[leak-audit] STACKED channels = %d  %s" % (len(names), list(names)))
        print("[leak-audit]   provenance    = %s"
              % sorted({f for c in names if c not in extra
                        for f in CHANNEL_PROVENANCE.get(c, ())}))
        print("[leak-audit]   truth fields  = %s -> NONE declared by any channel"
              % (list(TRUTH_FIELDS),))
        print("[leak-audit]   VIOLATIONS    = %s" % (rep["violations"] or "none"))
    if rep["violations"]:
        raise AssertionError("stacked-channel audit FAILED: %s" % rep["violations"])
    return rep


# =========================================================================== #
# 3.  ONE WELL, PACKED FOR THE ENCODER                                        #
# =========================================================================== #
class CoefSample:
    """One well: its input rows, its eval-grid size `n_eval`, and its basis.

    `ws` is a `rogii.seqwell.WellSeq`; only its INPUT fields are read
    (`base`, `extra`, `n`, `fe`, `name`).  `ws.tvt_true` / `ws.dtvt_true` -- the
    only fields WellSeq builds from `w.tvt` -- are never touched here.

    The two blocks are kept SEPARATE on purpose:
      `x`      base(7) + position(5).  Fixed units, pre-scaled by seqwell's
               FEAT_SCALE / this file's own constants -- no fitted statistic, so
               nothing here can depend on which wells are in the fold.
      `extra`  the feat_v6 / Z / gre2 channels, RAW.  They MUST be standardised
               per fold on the TRAINING wells only (`ChanNorm`); standardising
               them once over the whole corpus would fit a statistic on the
               validation wells, which is a leak.  `pack_batch(norm=...)` applies
               it at batch time so one `CoefSample` list serves all five folds.
    """
    __slots__ = ("name", "n", "fe", "n_eval", "x", "extra", "tn_eval", "rows",
                 "pos_names")

    def __init__(self, w, ws, pos_names=POS_CHANNELS, sub: int = EVAL_SUB):
        self.name = str(ws.name)
        self.n = int(ws.n)
        self.fe = int(ws.fe)
        self.pos_names = tuple(pos_names)
        rows, tn = eval_grid(w, sub)
        self.rows = rows
        self.tn_eval = tn.astype(np.float64)
        self.n_eval = int(len(rows))
        x = np.concatenate([np.asarray(ws.base, np.float32),
                            pos_channel_block(w, pos_names, sub)], 1)
        if not np.all(np.isfinite(x)):
            raise AssertionError("non-finite base/position channel in %s" % self.name)
        self.x = np.ascontiguousarray(x, np.float32)
        self.extra = (None if ws.extra is None else
                      np.ascontiguousarray(ws.extra, np.float32))
        if self.extra is not None and not np.all(np.isfinite(self.extra)):
            raise AssertionError("non-finite extra channel in %s" % self.name)

    @property
    def n_extra(self) -> int:
        return 0 if self.extra is None else self.extra.shape[1]

    @property
    def n_chan(self) -> int:
        return self.x.shape[1] + self.n_extra


def pack_batch(samples, device, norm=None):
    """Right-pad a list of CoefSample to one [B, T, C] tensor + [B, T] mask.

    `norm` is a fitted `rogii.seqwell.ChanNorm` for the `extra` block (fold's
    TRAINING wells only).  Passing None with a non-empty extra block is refused
    rather than silently feeding raw, mixed-unit channels through one LayerNorm.
    """
    B = len(samples)
    T = max(s.n for s in samples)
    C = samples[0].n_chan
    K = samples[0].n_extra
    if K and norm is None:
        raise AssertionError(
            "%d extra channels but no ChanNorm: fit it on the fold's TRAIN "
            "wells (a norm fitted on the val wells is a leak)" % K)
    if K and not getattr(norm, "fitted", False):
        raise AssertionError("ChanNorm.apply before fit -- that is the leak")
    x = np.zeros((B, T, C), np.float32)
    m = np.zeros((B, T), np.float32)
    tn = np.zeros((B, T), np.float32)
    ln = np.zeros(B, np.int64)
    for b, s in enumerate(samples):
        if s.n_chan != C:
            raise AssertionError("channel count differs inside a batch")
        j = s.x.shape[1]
        x[b, :s.n, :j] = s.x
        if K:
            x[b, :s.n, j:] = (s.extra - norm.mean[None, :]) / norm.std[None, :]
        m[b, :s.n] = 1.0
        tn[b, :s.n] = np.interp(np.arange(s.n, dtype=float),
                                s.rows.astype(float), s.tn_eval)
        ln[b] = s.n
    t = lambda a: torch.from_numpy(a).to(device)          # noqa: E731
    return dict(x=t(x), mask=t(m), tn=t(tn), lens=torch.from_numpy(ln), T=T)


# =========================================================================== #
# 4.  THE ENCODER                                                             #
# =========================================================================== #
class _DilBlock(nn.Module):
    """Residual dilated conv: the cheap way to a 1 000+ ft receptive field."""

    def __init__(self, d, dilation, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.c1 = nn.Conv1d(d, 2 * d, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv1d(2 * d, d, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, h, m):
        # h [B,T,D], m [B,T]
        z = self.ln(h).transpose(1, 2)
        z = self.c2(torch.nn.functional.gelu(self.c1(z))).transpose(1, 2)
        return (h + self.do(z)) * m.unsqueeze(-1)


class CoefNet(nn.Module):
    """Read the whole well, emit K numbers.

    stem     Linear -> LayerNorm, then a strided conv that takes the native 1 ft
             grid down to the `ds` ft grid the target's basis lives on (default
             4, i.e. the submission grid: docs/35 measured the 4 ft grid floor at
             0.0256 ft and 99.9992 % of the target's variance surviving, so the
             reduction is free).
    trunk    `blocks` residual dilated convs, dilations 1,2,4,8,... -- receptive
             field ~ ds * 3 * 2^blocks rows, i.e. >= 1 500 ft at the default, the
             scale of the ACF's L(1/e) = 462 ft (docs/25).
    rnn      BIDIRECTIONAL GRU over the reduced sequence, run through
             `pack_padded_sequence` so the backward pass never enters the pad.
    pool     masked mean + attention + the first three Legendre MOMENTS of the
             hidden state.  The moments are what makes an ODD output reachable:
             a symmetric pooling cannot represent `a1` at all.
    head     -> K numbers, LAST LAYER ZERO-INIT (so the untrained net is the BASE).

    `rot` [NCOEF, K] with orthonormal columns maps the K emitted numbers back to
    the 9 Legendre coefficients.  Identity slice = plain truncation; the docs/38
    Sec.5.7 population rotation is passed in by the trainer.  At K = 9 any
    orthogonal `rot` leaves the loss IDENTICALLY unchanged (it is an isometry of
    the coefficient space) -- asserted by the smoke, and the reason the default
    is `none`.
    """

    def __init__(self, n_chan: int, d_model: int = 64, ds: int = 4,
                 blocks: int = 5, gru_layers: int = 1, dropout: float = 0.1,
                 emit_k: int = NCOEF, head_hidden: int = 128,
                 rot: np.ndarray | None = None):
        super().__init__()
        self.n_chan = int(n_chan)
        self.d_model = int(d_model)
        self.ds = int(ds)
        self.emit_k = int(emit_k)
        self.proj = nn.Linear(n_chan, d_model)
        self.ln_in = nn.LayerNorm(d_model)
        self.stem = nn.Conv1d(d_model, d_model, 2 * ds + 1, stride=ds, padding=ds)
        self.blocks = nn.ModuleList(
            [_DilBlock(d_model, 2 ** i, dropout) for i in range(blocks)])
        self.gru = nn.GRU(d_model, d_model // 2, gru_layers, batch_first=True,
                          bidirectional=True,
                          dropout=dropout if gru_layers > 1 else 0.0)
        self.ln_out = nn.LayerNorm(d_model)
        self.attn = nn.Linear(d_model, 1)
        self.n_pool = 5                                   # mean, attn, P1, P2, P3
        self.head = nn.Sequential(
            nn.Linear(self.n_pool * d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, self.emit_k),
        )
        # the model STARTS on the gate it has to beat: ahat = 0 = do nothing.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        if rot is None:
            rot = np.eye(NCOEF, dtype=np.float64)[:, :self.emit_k]
        rot = np.asarray(rot, np.float32)
        if rot.shape != (NCOEF, self.emit_k):
            raise AssertionError("rot must be (%d, %d), got %s"
                                 % (NCOEF, self.emit_k, rot.shape))
        self.register_buffer("rot", torch.from_numpy(rot))

    # ---- pooling ----------------------------------------------------------
    def _pool(self, h, m, tn):
        den = m.sum(1, keepdim=True).clamp(min=1.0)
        hm = h * m.unsqueeze(-1)
        mean = hm.sum(1) / den
        s = self.attn(h).squeeze(-1).masked_fill(m < 0.5, -1e9)
        att = (h * torch.softmax(s, dim=1).unsqueeze(-1)).sum(1)
        t = tn
        parts = [mean, att]
        for pk in (t,
                   0.5 * (3.0 * t * t - 1.0),
                   0.5 * (5.0 * t ** 3 - 3.0 * t)):
            parts.append((hm * pk.unsqueeze(-1)).sum(1) / den)
        return torch.cat(parts, dim=-1)

    # ---- forward ----------------------------------------------------------
    def forward(self, batch: dict) -> torch.Tensor:
        x, m, tn = batch["x"], batch["mask"], batch["tn"]
        h = self.ln_in(self.proj(x)) * m.unsqueeze(-1)
        h = self.stem(h.transpose(1, 2)).transpose(1, 2)
        ds = self.ds
        md = m[:, ::ds][:, :h.shape[1]]
        if md.shape[1] < h.shape[1]:                       # odd lengths
            md = torch.nn.functional.pad(md, (0, h.shape[1] - md.shape[1]))
        tnd = tn[:, ::ds][:, :h.shape[1]]
        if tnd.shape[1] < h.shape[1]:
            tnd = torch.nn.functional.pad(tnd, (0, h.shape[1] - tnd.shape[1]),
                                          value=1.0)
        h = h * md.unsqueeze(-1)
        for blk in self.blocks:
            h = blk(h, md)
        lens = md.sum(1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lens, batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                total_length=h.shape[1])
        h = self.ln_out(h) * md.unsqueeze(-1)
        b = self.head(self._pool(h, md, tnd))              # [B, K]
        return b @ self.rot.t()                            # [B, 9] in FEET


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =========================================================================== #
# 5.  THE LOSS *IS* THE METRIC (docs/38 Q1 + Q2)                               #
# =========================================================================== #
def coef_loss(ahat: torch.Tensor, a: torch.Tensor, nw: torch.Tensor):
    """loss = sum_w n_w * sum_j (a_wj - ahat_wj)^2 / sum_w n_w.

    This is the pooled RMSE^2 MINUS the constant sum_w ||r_w||^2 / sum_w n_w
    (docs/38 Q1: 1.5806 % of the SSE, floor 1.6765 ft).  Same gradient, same
    argmin, and NOT a number of feet -- read `reconstructed_rmse`, never this.
    """
    d2 = ((a - ahat) ** 2).sum(dim=1)
    return (nw * d2).sum() / nw.sum().clamp(min=1.0)


def reconstructed_rmse(ahat, a, nw, sse) -> float:
    """THE gate quantity: the pooled per-row RMSE the submission would score.

        RMSE^2 = ( SSE_total - sum_w n_w * sum_j (2 ahat a - ahat^2) ) / sum n_w

    exact by intra-well orthogonality (docs/38 Sec.5.1 verifies it as an
    IDENTITY, 0.00e+00 ft over three noise regimes).  Feed `ahat = a` and you get
    the deg-8 projection floor; feed `ahat = 0` and you get BASE, to the bit.
    """
    ahat = np.asarray(ahat, float)
    a = np.asarray(a, float)
    nw = np.asarray(nw, float)
    g = float((nw * (2.0 * ahat * a - ahat ** 2).sum(1)).sum())
    return float(np.sqrt(max(float(np.asarray(sse, float).sum()) - g, 0.0)
                         / float(nw.sum())))


def r2_per_coef(ahat, a, nw):
    """R2_j = G_j / S_j, n-weighted and UNCENTRED (docs/38 Sec.5.1).

    This is the number that plugs straight into the docs/38 ceiling table: it can
    be NEGATIVE, and a negative R2_j means that coefficient is COSTING feet.
    """
    ahat = np.asarray(ahat, float)
    a = np.asarray(a, float)
    nw = np.asarray(nw, float)[:, None]
    S = (nw * a ** 2).sum(0)
    G = (nw * (2.0 * ahat * a - ahat ** 2)).sum(0)
    return G / np.maximum(S, 1e-30), S.ravel()


def permutation_null(ahat, a, nw, sse, nperm: int = 20, seed: int = 0):
    """docs/29 Sec.4 NULL: the SAME predicted coefficient vectors, handed to the
    WRONG wells.  Identical procedure, identical marginals, zero pairing.

    RETURNS THE DISTRIBUTION, not its mean.  The previous reading was
    `ok_null = rmse < null.mean()`, i.e. "better than the AVERAGE wrong answer",
    which is a coin flip: the refuter measured P(false positive) = 0.54 in
    general and 0.70 in the quasi-constant regime, where the null collapses
    (`refute_prodmode.json`: null_sd = 1.78e-15 on 1 well).  What is returned
    here supports the only honest readings:

      p05      the empirical 5th percentile -- the model must be BELOW it;
      p        (1 + #{null <= rmse}) / (nperm + 1), a valid permutation p-value
               (so `nperm >= 19` is REQUIRED for p <= 0.05 to be reachable);
      sd, ndistinct   how degenerate the null is.  A null with one distinct
               value is not a control, it is the model's own number echoed back.
    """
    rs = np.random.RandomState(seed)
    ahat = np.asarray(ahat, float)
    W = len(nw)
    v = np.array([reconstructed_rmse(ahat[rs.permutation(W)], a, nw, sse)
                  for _ in range(int(nperm))], float)
    rm = reconstructed_rmse(ahat, a, nw, sse)
    return dict(mean=float(v.mean()), sd=float(v.std()), min=float(v.min()),
                p05=float(np.quantile(v, 0.05)),
                p=float((1.0 + float((v <= rm).sum())) / (len(v) + 1.0)),
                nperm=int(len(v)), ndistinct=int(len(np.unique(np.round(v, 12)))),
                values=[float(x) for x in v])


def population_rotation(a, nw, k: int):
    """docs/38 Sec.5.7: the population's own orthonormal directions, TRAIN ONLY.

    Returns [9, k] with orthonormal columns, the top-k eigenvectors of the
    n-weighted uncentred second moment.  At k = 9 this is an isometry and the
    loss is unchanged; it only bites when k < 9 (measured out-of-fold: k=2 gives
    4.3727 against 5.1395 for the level+slope family, null 11.5316).
    """
    a = np.asarray(a, float)
    nw = np.asarray(nw, float)[:, None]
    M = (a * nw).T @ a
    lam, U = np.linalg.eigh(M)
    return np.ascontiguousarray(U[:, ::-1][:, :int(k)])


# =========================================================================== #
# 6.  RUNTIME PROOFS (they would otherwise fail SILENTLY)                     #
# =========================================================================== #
def truth_independence_check(w, ws_builder, pos_names=POS_CHANNELS,
                             sub: int = EVAL_SUB, extra_names=()):
    """T6, applied to this organ: destroy every truth field and the INPUT tensor
    must not move by one bit -- reported PER CHANNEL, not as one boolean.

    `ws_builder(well)` must return a WellSeq.  Also asserts the check is not
    vacuous, and again per channel: perturbing the GR log must move the GR
    channels specifically, so "nothing moved" and "everything is fine" stop
    being the same output.
    """
    import copy

    def stack(well):
        s = CoefSample(well, ws_builder(well), pos_names, sub)
        return (s.x if s.extra is None else
                np.concatenate([s.x, s.extra], 1))

    names = stacked_channel_names(pos_names, extra_names)
    ref = stack(w)
    if ref.shape[1] != len(names):
        raise AssertionError("tensor has %d channels, %d names declared"
                             % (ref.shape[1], len(names)))
    w2 = copy.copy(w)
    w2.tvt = np.full(int(w.n), 1234.5)
    w2.tops = {}
    w2.tw_geol = None
    w2.datum = 0.0
    got = stack(w2)
    moved_truth = [names[j] for j in range(ref.shape[1])
                   if not np.array_equal(ref[:, j], got[:, j])]
    if moved_truth:
        raise AssertionError(
            "channel(s) MOVED when w.tvt/tops/tw_geol/datum were destroyed -- "
            "THE TRUTH IS ON THE INFERENCE PATH: %s" % moved_truth)
    w3 = copy.copy(w)
    w3.gr = np.asarray(w.gr, float) * 1.7 + 9.0
    alt = stack(w3)
    moved_gr = [names[j] for j in range(ref.shape[1])
                if not np.array_equal(ref[:, j], alt[:, j])]
    if not moved_gr:
        raise AssertionError("inputs are insensitive to GR: the check is vacuous")
    return dict(rows=int(ref.shape[0]), chan=int(ref.shape[1]),
                names=list(names), moved_on_truth=moved_truth,
                moved_on_gr=moved_gr, vacuous=False)


# --------------------------------------------------------------------------- #
# 6b.  THE TWO CHANNEL CENSUSES                                                #
#                                                                              #
# A GRADIENT census counts PARAMETER TENSORS.  It cannot see an inert CHANNEL:  #
# every parameter of `proj` is one tensor, so a column of `proj.weight` that    #
# multiplies a constant channel still receives a non-zero gradient through its  #
# other columns and the tensor is reported "fed".  That is exactly how the      #
# 12th channel (`known`, an exact affine copy of `is_eval`) survived a smoke     #
# that printed "40/40 tensors fed".  Two independent censuses are needed:       #
#   DATA    -- does the channel carry any variation, and is it a copy?          #
#   WIRING  -- does perturbing the channel MOVE the model's output?             #
# --------------------------------------------------------------------------- #
def duplicate_channels(X, names, tol: float = 1e-9):
    """Pairs (i, j) where channel j is an EXACT affine function of channel i.

    Least squares on the pooled rows, in float64, then the max residual relative
    to the channel's own sd.  `known` = 1 - `is_eval` gives residual 0.000e+00.
    """
    X = np.asarray(X, np.float64)
    out = []
    sd = X.std(0)
    for i in range(X.shape[1]):
        if sd[i] <= 0:
            continue
        for j in range(i + 1, X.shape[1]):
            if sd[j] <= 0:
                continue
            A = np.stack([X[:, i], np.ones(len(X))], 1)
            c, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
            r = float(np.abs(A @ c - X[:, j]).max()) / max(sd[j], 1e-30)
            if r <= tol:
                out.append((names[i], names[j], float(r), float(c[0]), float(c[1])))
    return out


def channel_census(samples, names, eval_masks=None, max_rows: int = 400000):
    """DATA census: per channel, what the encoder actually receives.

    Columns: sd over all rows / over the EVALUATED span / over the heel, the
    fraction of exact zeros on the evaluated span, and p99.9 |x|.  Raises when a
    channel is dead over the WHOLE well (sd == 0 everywhere -- it cannot be
    anything but bias), when a channel is an exact affine copy of another, or
    when a channel exceeds MAX_CHANNEL_AMPLITUDE.

    "dead on the evaluated span but alive on the heel" is REPORTED, never fatal:
    `heel_off` and `dtvt_in` are exactly that by construction (TVT_input does not
    exist past the PS) and the encoder pools over the whole well.
    """
    rows, ev = [], []
    tot = 0
    step = max(1, len(samples) // 40)          # spread over the corpus, not its head
    for k in range(0, len(samples), step):
        s = samples[k]
        x = s.x if s.extra is None else np.concatenate([s.x, s.extra], 1)
        if eval_masks is None:
            m = np.zeros(s.n, bool)
            m[s.rows] = True
        else:
            m = np.asarray(eval_masks[k], bool)
        rows.append(np.asarray(x, np.float64))
        ev.append(m)
        tot += len(x)
        if tot >= max_rows:
            break
    X = np.concatenate(rows, 0)
    EV = np.concatenate(ev, 0)
    if X.shape[1] != len(names):
        raise AssertionError("census: %d channels, %d names" % (X.shape[1], len(names)))
    HEEL = ~EV
    # the heel is "everything not on the eval grid"; use is_eval when present
    if "is_eval" in names:
        HEEL = X[:, list(names).index("is_eval")] < 0.5
        EVSPAN = ~HEEL
    else:
        EVSPAN = EV
    out = []
    for j, nm in enumerate(names):
        v = X[:, j]
        out.append(dict(
            name=nm, sd_all=float(v.std()),
            sd_eval=float(v[EVSPAN].std()) if EVSPAN.any() else 0.0,
            sd_heel=float(v[HEEL].std()) if HEEL.any() else 0.0,
            zero_frac_eval=float((v[EVSPAN] == 0).mean()) if EVSPAN.any() else 1.0,
            amp=float(np.percentile(np.abs(v), 99.9)), mean=float(v.mean())))
    dead = [d["name"] for d in out if d["sd_all"] == 0.0]
    dup = duplicate_channels(X, list(names))
    over = [(d["name"], d["amp"]) for d in out if d["amp"] > MAX_CHANNEL_AMPLITUDE]
    viol = []
    if dead:
        viol.append("channel(s) CONSTANT over every row of every well (they are "
                    "the bias, nothing else): %s" % dead)
    if dup:
        viol.append("channel(s) that are an EXACT affine copy of another "
                    "(zero bits): %s" % [(a, b) for a, b, *_ in dup])
    if over:
        viol.append("channel(s) over the p99.9 amplitude ceiling %.1f: %s"
                    % (MAX_CHANNEL_AMPLITUDE, over))
    return dict(rows=int(len(X)), n_wells=int(len(rows)), per_channel=out,
                duplicates=[list(d) for d in dup], violations=viol,
                heel_only=[d["name"] for d in out
                           if d["sd_eval"] == 0.0 and d["sd_heel"] > 0.0])


def wiring_census(net, batch, names, device=None, delta: float = 1.0,
                  seed: int = 12345):
    """WIRING census: perturb ONE channel, the output must MOVE.

    This is the check a gradient census cannot do.  It needs a NON-ZERO head --
    with the zero-init head every output is identically 0 and every channel would
    look dead -- so the head's last layer is temporarily given a fixed random
    draw (seed `seed`) and restored afterwards.  Returns max |d ahat| per channel
    and raises if any channel moves the output by exactly nothing.
    """
    import torch as _t

    head = net.head[-1]
    keep_w = head.weight.detach().clone()
    keep_b = head.bias.detach().clone()
    g = _t.Generator(device="cpu").manual_seed(int(seed))
    if float(keep_w.abs().sum()) == 0.0:
        head.weight.data.copy_(_t.randn(keep_w.shape, generator=g) * 0.1)
        head.bias.data.copy_(_t.randn(keep_b.shape, generator=g) * 0.1)
    try:
        net.eval()
        with _t.no_grad():
            ref = net(batch).cpu().numpy().astype(np.float64)
            moved = []
            for j in range(batch["x"].shape[2]):
                b2 = dict(batch)
                x2 = batch["x"].clone()
                x2[:, :, j] = x2[:, :, j] + delta * batch["mask"]
                b2["x"] = x2
                d = float(np.abs(net(b2).cpu().numpy().astype(np.float64)
                                 - ref).max())
                moved.append(d)
    finally:
        head.weight.data.copy_(keep_w)
        head.bias.data.copy_(keep_b)
    inert = [names[j] for j, d in enumerate(moved) if d == 0.0]
    if inert:
        raise AssertionError(
            "channel(s) WIRED BUT INERT -- perturbing them by %.3g moves the "
            "output by EXACTLY zero: %s" % (delta, inert))
    return dict(delta=float(delta),
                per_channel={names[j]: float(moved[j]) for j in range(len(names))},
                inert=inert)


def basis_identity_check(tn: np.ndarray, e: np.ndarray, dmax: int = DMAX):
    """The algebra, on one well, from first principles.

    Returns the worst |LHS - RHS| over a few arbitrary `ahat`, where
        LHS = ||e - sum_j ahat_j P_j||^2
        RHS = ||r||^2 + n * sum_j (a_j - ahat_j)^2
    docs/38 Q1 measures RHS exact to 6.3e-09 ft^2 and the naive form (RHS without
    ||r||^2) wrong by up to 3.28e+04 ft^2.
    """
    n = len(tn)
    Q = ortho_basis(tn, dmax)
    P = Q * math.sqrt(n)
    a = (Q.T @ e) / math.sqrt(n)
    rr = float(e @ e) - n * float(a @ a)
    rs = np.random.RandomState(1)
    worst_true, worst_naive = 0.0, 0.0
    for _ in range(5):
        ah = a + rs.normal(0, 1.0, dmax + 1) * np.abs(a).mean()
        lhs = float(((e - P @ ah) ** 2).sum())
        gap = n * float((a - ah) @ (a - ah))
        worst_true = max(worst_true, abs(lhs - (rr + gap)))
        worst_naive = max(worst_naive, abs(lhs - gap))
    return dict(a=a, rr=float(rr), exact=float(worst_true),
                naive=float(worst_naive))
