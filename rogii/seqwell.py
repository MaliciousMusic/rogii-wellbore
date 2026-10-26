"""rogii.seqwell -- ONE WELL = ONE SAMPLE autoregressive sequence model.

This is the SECOND, independent route to the toe TVT, built to the published S6
specification (crabxmz, docs/21_corpus_mining.md Sec.5) rather than to a hand-built
energy.  Nothing here is a decoder over a hand-written posterior: the increment is
produced by a GRU that is trained through its own rollout.

WHAT THE SPEC FIXES (docs/21_corpus_mining.md Sec.5, "S6"):
  * target      = the per-step INCREMENT relative to -dZ.
                  TVT[i] = TVT[i-1] + (-dZ[i] + r[i]) ; the net learns only r.
  * base feats  = d_X, d_Y, d_Z, d_TVT, d_GR_n, GR_n   (GR normalised by the
                  TYPEWELL's mean/std so both logs share one scale)
  * profile     = 29-dim multi-scale typewell correlation profile:
                  tw_gr_n(t_hat + off_j) - gr_n[i]   for 29 offsets,
                  dense -10..+10 plus coarse out to +-40.
  * model       = 2-layer GRU, hidden 128, dropout 0.1, 256-step windows.
                  Head = [last GRU state ; NEXT step's profile] -> 2-layer MLP -> r.
  * norm        = LayerNorm on the input.  NEVER BatchNorm (the net runs (B,T,C)
                  in training and (B,C) in rollout; BatchNorm running stats get
                  polluted by rollout inputs).
  * phase 1     = teacher forcing on a track corrupted by a clipped cumulative
                  random walk, step sigma = 0.08 clipped to +-15.  Targets stay TRUE.
                  Loss = 50/50 blend of delta-RMSE and absolute-TVT RMSE.
  * phase 2     = backprop through genuine K-step rollouts with a differentiable
                  torch.gather typewell lookup.  K = 16 -> 64 -> 256.
  * checkpoint  = full-well ROLLOUT MAE on a fixed validation set.  NEVER the
                  teacher-forced validation loss ("it actively picked bad models").
  * inference   = encode the known prefix anchored on TVT_input, then step; wherever
                  TVT_input is known the running value SNAPS BACK to the anchor.

TEST SAFETY.  Every input is the well's own MD/X/Y/Z/GR/TVT_input and its own
typewell TVT/GR.  `w.tvt`, `w.tops`, `w.tw_geol`, `w.datum` are NEVER touched on
any path in this file.  There is exactly one place the truth enters -- the
training TARGET (`WellSeq.tvt_true` / `WellSeq.dtvt_true`, built from `w.tvt`) --
it is None whenever the well has no truth, and it is only ever consumed by the
loss and by the scorer.  `predict_full` reads base / dz / grn / tvt_in / known
and nothing else.

=============================================================================
DEFAULTS THE SOURCE DOES NOT SPECIFY.  Each one is MARKED `[UNSPEC]` here and is
a free hyper-parameter, exactly like the 27-row register in docs/20_stride_spec.md.
=============================================================================
U1  [UNSPEC] The 8 COARSE offsets.  The doc says "coarse points out to +-40",
    not which.  Default +-{15, 22, 30, 40} -- 15 and 22 straddle the measured
    15-25 ft bedding repeat, 30/40 give the two aliases out.
U2  [UNSPEC] Typewell resampling grid step.  Doc says "0.25-unit gridded"; the raw
    typewells here are on 0.5 ft, so 0.25 is an interpolation.  TW_DT = 0.25.
U3  [UNSPEC] Grid padding beyond the typewell TVT span: TW_PAD = 60 ft (max offset
    40 + margin).  np.interp edge-clamps, so the profile saturates rather than
    exploding when the track leaves the typewell.
U4  [UNSPEC] GR normalisation is the typewell's mean/std as written.  A heel-fitted
    affine recalibration (gain 0.783 / offset +17.6 measured by diag_stride) is
    available as `gr_heel_affine=True` and is OFF by default (spec fidelity).
U5  [UNSPEC] Fixed pre-scales on the base features (FEAT_SCALE) so that a single
    LayerNorm across 37 mixed-unit channels does not wash out dZ ~ 0.05 against a
    profile of order 1.  This is a per-channel constant, not a learned/batch stat.
U6  [UNSPEC] `res_scale` = 0.05: the head's linear output is multiplied by it, so a
    zero-init head starts the model EXACTLY on the structural prior TVT = -Z + c.
U7  [UNSPEC] The 50/50 loss blend needs units.  delta-RMSE is O(0.02 ft) and
    absolute-TVT RMSE is O(3 ft), so a literal sum is 100% the absolute term.
    Both are divided by a fixed scale (delta_scale=0.02, abs_scale=3.0) first.
U8  [UNSPEC] Rollout warm-up length before the K-step window: warmup = 64 steps,
    teacher-forced on the noisy track, hidden state detached at the hand-over.
U9  [UNSPEC] Phase-2 loss = the same 50/50 blend, on the ROLLED-OUT absolute TVT
    and its deltas (the doc only fixes the checkpoint criterion, not the loss).
U10 [UNSPEC] Two extra base channels beyond the doc's six: d_MD (constant 1.0 ft
    here, kept for portability) and the `gr_obs` mask (GR is 52-91% observed per
    well and dataset.py interpolates the holes -- the net must be able to tell).

=============================================================================
M1.  THE SPEC'S STRUCTURAL BASE IS REFUTED ON THIS DATASET.  MEASURED, 86 wells,
     414 439 eval rows, pure arithmetic -- no model, no fit, no search:

         TVT_pred = anchor - beta * (Z - Z_anchor)      pooled per-row RMSE
             beta = 0.00  (hold-last)      15.4949
             beta = 0.10                   18.2761
             beta = 0.20                   25.4893
             beta = 0.30                   34.4552
             beta = 0.50                   54.1052
             beta = 0.75                   79.6873
             beta = 1.00  (SPEC)          105.6352

     Monotone.  The spec writes "flat formation => dTVT = -dZ" and bases the whole
     increment on it; here that base is 6.8x WORSE than holding the last value.
     Cause, measured on the same wells: over the toe, Z moves 100-260 ft while TVT
     moves 12-66 ft.  The driller STEERS TO STAY IN ZONE, so the well's vertical
     motion IS the formation dip -- and the dip cancels it.

     The decomposition, measured on 20 wells:
         sd(dTVT)        = 0.21 .. 0.35 ft/step
         sd(dTVT + dZ)   = 0.022 .. 0.12 ft/step        (about 10x smaller)
         sum(dZ) over toe = +100..+263 ft   vs   sum(dTVT) = -10..-47 ft
     So -dZ is an EXCELLENT high-pass prior -- it removes ~90% of the per-step
     variance -- and a DISASTROUS integrator: what it leaves is not noise but a
     persistent mean of about 0.022 ft/step (the dip), and 4 700 toe steps of that
     is the 90-230 ft of drift.  Step-to-step corr(dTVT, dZ) = -0.57..-0.79 with
     slope -0.45..-1.25; cumulatively the relation is gone.

     This is the same shape as the STRIDE failure and worth stating once: the
     metric is on the INTEGRAL, so a prior that is right locally and biased
     globally is worse than no prior at all.

     Consequence for the build: `prior_beta` is a first-class parameter of the
     update  TVT[i] = TVT[i-1] - prior_beta*dZ[i] + r[i], the head is zero-init so
     an untrained net reproduces that prior EXACTLY (verified: |net - analytic|
     <= 0.51 ft after 4 000 float32 rollout steps), and the default is
     **prior_beta = 0.0**, i.e. the net's residual IS dTVT.  `--prior-beta 1.0`
     restores literal spec fidelity.  dZ remains an input channel either way, so
     the net can still learn any beta it wants -- what changes is only where the
     zero-init model starts, which under a short budget is most of the outcome.
=============================================================================
M2.  EXTRA PER-ROW CHANNELS  (FRONT G, added 2026-07-26).  OFF BY DEFAULT --
     with `feat_cols=()` and `z_cols=()` the input is bit-identically the 37-dim
     tensor above and every running fold stays reproducible.

     WHY.  Three independent measurements point the same way:
       * docs/21 (corpus): a from-scratch aligner generalises at ~14 ft and only
         beats the hand-feature ceiling when the ~50 engineered features are fed
         in ALONGSIDE the learned cost volume (10.34).  620 wells is data-starved.
       * docs/25: rho(seqwell remainder, Z) = -0.42 (LGBM: -0.055) => seqwell does
         NOT use the known trajectory, and a single global Z coefficient is worth
         -0.44 ft on its body.
       * docs/25: in the body seqwell's remainder sits at rho 0.955 with hold-last
         => it adds almost no shape there.  Its measured value is tail decorrelation.

     TWO HARD CONSTRAINTS THE DESIGN OBEYS, both measured:
       * docs/29: the 73 AGGREGATED per-well descriptors `w_*` are worth EXACTLY
         ZERO (R2 -0.0038, below null); all the signal lives in the 680
         along-the-toe statistics.  => channels enter PER ROW, never as a summary
         vector.  Enforced mechanically: `assert_varying()` refuses any column
         whose within-well sd is below `MIN_FRAC_WITHIN` of its total sd.
         RE-MEASURED here (`tools/seq_channel_select.py`, 946 281 rows / 773
         wells): **88 of the 210** feat_v6 columns are per-well constants and are
         rejected by it; the eligible 122 sit at frac_within >= 0.17.
       * docs/30: every heel-anchored feature decays along the toe (`dz`: |corr|
         0.491 -> 0.113 across deciles).  => candidates are ranked by `far3`, the
         mean |corr(col, u)| over frac_eval deciles 7-9 with u = y_tvt - _tvt_last
         (docs/30 Sec.0), not by LGBM gain.  The whole ranking is re-measured by
         `tools/seq_channel_select.py` and lives in
         `data/derived/_seqg/chan_select.csv`; every far3 quoted below is from
         that file, not from a doc.  The ranking is a WITNESS, not a proof: it is
         a linear screen on a single-well-pooled correlation, so it can only
         reject, never promote.  The verdict is the fold RMSE.

     ALIGNMENT.  feat_v6 lives on `np.arange(w.fe, w.n)[::sub]` -- toe rows only,
     1-in-4.  The model runs on `np.arange(w.n)` -- heel included.  Each column is
     mapped with `np.interp` on the ROW INDEX: exact at the sampled rows, linear
     between them, and EDGE-CLAMPED outside (so every heel row carries the value
     of the first sampled toe row, a legal test-safe quantity, and no fabricated
     variation).  NaN are dropped before the interpolation; a column that is
     all-NaN for a well becomes the constant 0 and is COUNTED and PRINTED.

     NORMALISATION.  Per channel, mean/sd over the TRAINING WELLS ONLY, stored in
     the checkpoint (`ChanNorm`).  A channel standardised with validation/test
     statistics is a leak.  The net still sees only LayerNorm -- never BatchNorm.
=============================================================================
M3.  THE gre2 EVIDENCE CHANNELS  (FRONT J, added 2026-07-26).  OFF BY DEFAULT.

     WHAT.  `rogii/gre2.py` is the screened extension of `gre_off_mean` -- the
     ONE feature family whose informativeness RISES along the toe (docs/30 F5).
     docs/37 screened 168 variants on 200 fold-balanced wells x 2 seeds against
     TWO gates (beat the incumbent BASE *and* beat its own typewell-shuffle
     NULL) and kept 16, of which the ones wired here are in `GRE2_SETS`.

     ROUTE: NATIVE, not a parquet.  The gre2 columns do NOT exist in
     feat_v6_train.parquet.  There were two ways to get them to the net:
       (a) materialise them into a parquet keyed (well,row) like feat_v6, so
           `FeatSource` + `align_feats` pick them up unchanged;
       (b) compute them per well inside `WellSeq`, like the Z channels.
     (b) is what this file does.  The trade, stated honestly:
       + (a) is simpler and reuses the already-audited alignment path.
       - (a) needs a 946 281-row build over 773 wells, AND a SECOND build for
         the test split, AND a bit-parity check between the two -- the exact
         surface that `docs/README` Sec.4 records as having caught real bugs.
         `tools/seqwell_kernel.py` calls `build_wellseq(w, p=P)` with
         `feats=None`, so a parquet-backed channel silently becomes the constant
         0 at submission time (`feat_missing`).  A natively-computed channel is
         simply there, on train and on test, with no second artefact to keep in
         sync.
       - (a) is stuck on feat_v6's 1-in-4 toe grid and reaches the model's
         native rows only through `np.interp`; gre2's evidence is an
         ACCUMULATION along the rows it is given, so (b) accumulates it at the
         native 1 ft spacing over 4x as many GR samples.
       - (a) freezes the channel against the REAL typewell, so in the shuffle
         control it can only ever be zeroed.  (b) additionally RE-COMPUTES it
         against the donor typewell (`WellSeq(tw=...)`), so the control is
         honest twice over.  It is still zeroed -- see the next paragraph.
       - COST is not the deciding factor either way: MEASURED 0.076-0.153 s per
         well (5 wells, 4 762-8 568 rows, CPU, the 3 specs `gre2a` needs) at the
         native 1 ft grid, i.e. ~1.6e-5 s/row and ~95 s for all 773 wells --
         about the same as reading them off disk.  An opt-in disk cache exists
         (`gre2_channel_block(cache_dir=...)`) and is keyed by a CONTENT hash,
         never by well name: the shuffle arm computes the same well against a
         DIFFERENT typewell, so a name-keyed cache would hand the control the
         real well's evidence back -- the precise failure this whole wiring is
         supposed to prevent.

     SHUFFLE NEUTRALISATION IS MANDATORY.  Every gre2 channel is computed
     AGAINST the typewell, so all of them carry the `gre2_` prefix, which is in
     `TW_DERIVED_PREFIXES`, so `SeqParams.tw_derived_mask()` marks them and
     `ChanNorm.apply(neutral=...)` writes them to exactly 0 (= the train mean)
     in the control.  If they were not zeroed the control would read the real GR
     evidence straight back in and UNDERSTATE the channel -- the same trap the
     far15 wiring already documents.  `tools/train_seqwell.py` asserts that every
     gre2 channel is in that mask rather than trusting the prefix table.

     WHAT IS REFUSED, BY NAME, WITH ITS MEASUREMENT (docs/37 Sec.5):
       * `D_pseudo_*` / `F_pseudolog_*` -- NULL CONTAMINATED.  The pseudo-
         typewell blends the well's OWN heel into the reference and
         `shuffled_typewell` cannot destroy that: `F_pseudolog_w0_mean`'s
         partial correlation (0.206) sits BELOW its own partial-shuffle (0.228).
         The mandatory control is not valid for that family, so it is not wired.
       * every `_arg` / `_ent` / `_peak` / `_gap` / `_gapoff` / `_drift` --
         0 of 144 passed the BASE gate on either seed, and `_ent` (0.146 vs null
         0.211) and `_gap` (0.054 vs 0.065) are UNDER their own null: they
         describe the geometry of the offset grid, not the stratigraphy (the
         same verdict as P11).
       * every `_w200` -- dominated everywhere by w0 or w500.
     Both refusals are enforced in `audit_channels`, not left to the caller.

     KEEP THE INCUMBENT.  r(E_callog_w0_mean, gre_off_mean) = 0.75 in the far
     deciles, so 43 % of E's variance is new and 57 % is not: gre2 EXTENDS
     `gre_off_mean`, it does not replace it.  The intended arms combine the two
     (`gre1_gre2a`, `far15_gre2a`).

     STILL A SCREEN.  docs/37 Sec.0: no measurement in it is a gain in feet, and
     docs/29 F7 measured a channel that beat its null by -0.49 ft while being
     +0.12 ft WORSE than doing nothing.  The pre-registered verdict is an A/B on
     >= 150 fold-balanced wells, out-of-fold, against BASE and against NULL.
=============================================================================
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import time
from dataclasses import dataclass, field, replace as _dc_replace

import numpy as np
import torch
import torch.nn as nn

from rogii import gre2 as _gre2

# --------------------------------------------------------------------------- #
# the 29-dim multi-scale offset ladder                                         #
# --------------------------------------------------------------------------- #
DENSE_OFF = np.arange(-10.0, 10.5, 1.0)                       # 21 points, 1 ft
COARSE_OFF = np.array([-40.0, -30.0, -22.0, -15.0,            # U1
                       15.0, 22.0, 30.0, 40.0])               # 8 points
OFFSETS = np.sort(np.concatenate([DENSE_OFF, COARSE_OFF])).astype(np.float32)
NPROF = len(OFFSETS)
assert NPROF == 29, NPROF

TW_DT = 0.25          # U2
TW_PAD = 60.0         # U3

# base channels: dMD, dX, dY, dZ, GR_n, dGR_n, gr_obs           (U10 for 1 and 7)
N_BASE = 7
FEAT_SCALE = np.array([1.0, 1.0, 1.0, 20.0, 1.0, 1.0, 1.0], np.float32)   # U5
DTVT_SCALE = 20.0                                                          # U5


# =========================================================================== #
# M2 -- extra per-row channels                                                #
# =========================================================================== #
DERIVED = os.environ.get("ROGII_DERIVED",
                         r"C:\Users\agent\Desktop\ROGII\data\derived")
FEAT_PARQUET = os.path.join(DERIVED, "feat_v6_train.parquet")

# Columns that must never reach an inference path.  The first five are the
# feat_v6 trailing block (`y_delta` / `y_tvt` ARE the truth); the prefixes are the
# project's canonical cross-well list, copied verbatim from
# tools/build_features_v6.py so the two cannot drift apart silently.
BANNED_COLS = ("y_delta", "y_tvt", "well", "row", "_tvt_last")
CROSS_WELL_PREFIXES = (
    "pred_s_", "ds_w", "ds_idw", "d_w", "sd_w", "neff", "s_med", "pred_surf",
    "n_wells", "beh_", "w_biasstd", "w_biastrend", "w_biaslate", "trend400",
    "dip_", "pred_dip", "w_bias", "w_surf", "log_d_w", "form_pos_surf_med",
    "ds_med", "dipn", "int_dipn", "w_dipbias", "w_dip_heel",
)
# Channels whose value is computed AGAINST THE TYPEWELL.  In the shuffle control
# the typewell is replaced by a donor's; if these carried the real well's values
# the control would leak the GR channel straight back in.  Tagged so the control
# can neutralise them (see ChanNorm.apply(neutral=...)).
TW_DERIVED_PREFIXES = ("gre_", "gre2_", "agg_", "beam_", "ncc_", "wls_", "tw_",
                       "w_tvt_", "w_u_", "grx_tw_")
# `gre2_` is listed EXPLICITLY even though it looks like a `gre_` match: it is
# not one ("gre2" vs "gre_"), and every gre2 channel MUST be neutralised in the
# typewell-shuffle control because all of them are computed against the typewell.

# docs/29 gate: a column whose within-well sd is below this fraction of its total
# sd is a per-well summary in disguise.  Measured on feat_v6_train: 80 of the 210
# columns sit at frac_within < 1e-5, the other 130 at > 0.17.  The threshold is in
# the empty middle of that gap, so it is not a tuned number.
MIN_FRAC_WITHIN = 0.01

# --------------------------------------------------------------------------- #
# THE SELECTED SETS.  `far15` is the default when channels are switched on.
# Every member was chosen from ONE measured table and nothing else:
#   data/derived/_seqg/chan_select.csv, written by tools/seq_channel_select.py
#   over the whole feat_v6_train matrix (210 columns x 946 281 rows x 773 wells).
#     far3      = mean |corr(col, u)| over frac_eval deciles 7,8,9, u = the true
#                 correction from the anchor y_tvt - _tvt_last  (docs/30 Sec.0)
#     near3     = the same over deciles 0,1,2
#     survives  = |corr|(bin9) >= |corr|(bin0), the docs/30 admission criterion
#     fw        = frac_within, the docs/29 per-row gate (>= 0.01 required)
#     gain      = LGBM gain rank in runs/v5_nospatial_clean/importance.csv;
#                 blank = the column is not in the v5 feature list at all, which
#                 is the point -- those are channels the LB champion never saw.
# The screen is LINEAR and POOLED, so it is a REJECTOR, not a promoter: a column
# with far3 ~ 0 cannot help linearly, but a high far3 is not evidence of value.
# The only verdict is the fold RMSE, against BASE and against NULL (docs/29).
# --------------------------------------------------------------------------- #
FEAT_SETS = {
    "none": (),
    # the single docs/30 gift, alone -- the minimal ablation arm
    "gre1": ("gre_off_mean",),
    # the four best far3 channels in the whole matrix
    "far4": ("gre_off_mean", "agg_beam_minus_gre", "agg_ncc_minus_gre",
             "agg_signal_mean"),
    # the 8 best far3, no mechanism balancing -- the "rank alone" arm
    "far8": ("gre_off_mean", "agg_beam_minus_gre", "agg_ncc_minus_gre",
             "agg_signal_mean", "gre_entropy", "agg_signal_med",
             "beam_tvt_vloose", "agg_gre_abs"),
    # DEFAULT.  15 channels, one per mechanism, all with within-well variation.
    "far15": (
        # -- GR-vs-typewell evidence (the only family whose informativeness RISES
        #    with distance from the heel, docs/30 4b) --------------------------
        "gre_off_mean",         # far3 .3239 near3 .2387 .185->.284 surv fw .17 gain #2
        "gre_off_arg",          # far3 .0986 surv fw .72 gain #101   (mode vs mean)
        "gre_entropy",          # far3 .1860 .038->.173 surv fw .39 gain #19 (sharpness)
        # -- inter-family DISAGREEMENT (uncertainty, not depth) ---------------
        "agg_beam_minus_gre",   # far3 .2977 .162->.257 surv fw .21   <- #2 of 210
        "agg_ncc_minus_gre",    # far3 .2621 flat (.264->.244, NOT surv) fw .42
        "agg_gre_abs",          # far3 .1500 .033->.134 surv fw .24
        "agg_signal_mean",      # far3 .2428 .050->.203 surv fw .42 (4-channel consensus)
        "agg_signal_std",       # far3 .0921 .000->.081 surv fw .56 (its spread)
        # -- independent depth estimators (different mechanism, same target) --
        "beam_tvt_vloose",      # far3 .1591 (.252->.151, NOT surv) fw .51  best beam
        "beam_drift_stiff",     # far3 .0630 surv HARD (.0011 -> .0831) fw .94
        "ncc_lag_h8",           # far3 .1156 .095->.142 surv fw .79  short-window lag
        "ncc_lag_h25",          # far3 .1291 .111->.161 surv fw .78  long-window lag
        # -- GR texture: a family seqwell has NO access to today --------------
        "grx_rms_601",          # far3 .0953 .043->.161 surv fw .78
        "grx_cum_absgrad",      # far3 .0830 .043->.114 surv fw .53
        # -- FORWARD-looking trajectory: the one channel that is not heel-anchored
        "dls_fwd_1000",         # far3 .0705 .038->.065 surv fw .87 gain #78
    ),
    # the upward sweep arm: far15 + the next 10 eligible SURVIVORS by far3.
    # 25 channels is the top of the range 620 wells can plausibly carry.
    "far25": (
        "gre_off_mean", "gre_off_arg", "gre_entropy",
        "agg_beam_minus_gre", "agg_ncc_minus_gre", "agg_gre_abs",
        "agg_signal_mean", "agg_signal_std",
        "beam_tvt_vloose", "beam_drift_stiff", "ncc_lag_h8", "ncc_lag_h25",
        "grx_rms_601", "grx_cum_absgrad", "dls_fwd_1000",
        "agg_signal_med",        # far3 .1722 .011->.141 surv fw .53
        "ncc_lag_h15",           # far3 .1257 .108->.154 surv fw .78
        "ncc_score_mean",        # far3 .0756 .032->.075 surv fw .88
        "agg_signal_range",      # far3 .0783 .009->.069 surv fw .57
        "agg_pair_absdiff_max",  # far3 .0725 .007->.065 surv fw .56
        "dls_fwd_2500",          # far3 .0679 .032->.065 surv fw .79 gain #12
        "grx_env_601",           # far3 .0669 .070->.130 surv fw .77
        "grx_rms_201",           # far3 .0609 .040->.122 surv fw .85
        "beam_drift_vcons",      # far3 .0476 .0007->.073 surv fw .95
        "ncc_score_h25",         # far3 .0661 .037->.064 surv fw .89
    ),
}
# DELIBERATE EXCLUSIONS, each with its measurement (chan_select.csv):
#  * every `w_*` (73 cols) and every `wls_b/slope/affine/resid/half` -- per-well
#    CONSTANTS.  88 of 210 columns measure frac_within < 0.01 and are refused
#    mechanically by `FeatSource.assert_varying`.  docs/29: worth exactly zero.
#  * `frac_eval` -- docs/30 5a measured the whole family of frac_eval-dependent
#    corrections at -0.005 ft against a -0.005 null, oracle ceiling -0.048.
#  * `gre_sd`, `gre_peak`, `gr_off` -- docs/30 5d measured all three NEGATIVE
#    (+0.0006 / +0.0005 / +0.0039); `gre_off_mean` is an isolated case, not a
#    family representative.
#  * `dz`, `dz_rate` -- the textbook heel-anchored decay (near3 .364 / .355 vs
#    far3 .116 / .125, both FAIL `survives`).  NOT excluded but MOVED into the Z
#    GROUP below, computed natively so they exist on the heel rows too and so
#    their contribution is separately switchable (the docs/25 pre-registered gate).

# --------------------------------------------------------------------------- #
# Z channels.  Computed natively from w.md / w.z / w.fe -- NOT read from the
# parquet -- so they exist on every row including the heel and do not depend on
# feat_v6 being built for the split.  Z is known everywhere, train AND test.
# docs/25 measured rho(seqwell remainder, Z) = -0.42 => the net ignores the
# trajectory it is given.  Pre-registered gate: >= -0.30 ft on >= 2 folds, with
# a permuted-Z control < 0.05 and a positive typewell-shuffle control.
# --------------------------------------------------------------------------- #
Z_SETS = {
    "none": (),
    "all": ("zcum", "zrate", "zdisp_rate", "zperp"),
    "cum": ("zcum",),
    "rate": ("zrate",),
}
Z_CHANNELS = ("zcum", "zrate", "zdisp_rate", "zperp")


def z_channel_block(md, z, fe, names) -> np.ndarray:
    """The Z-derived per-row channels, in the model's own units (TVT feet).

    Reads md, z, fe and NOTHING else.  Sign convention follows TVT = -Z + S + T_w,
    so a rise in Z lowers TVT and each channel is written in the TVT sense.
    """
    md = np.asarray(md, float)
    z = np.asarray(z, float)
    n = len(md)
    a = max(int(fe) - 1, 0)
    z0, md0 = float(z[a]), float(md[a])
    dmd = np.zeros(n); dmd[1:] = np.diff(md)
    dz = np.zeros(n); dz[1:] = np.diff(z)
    dm = md - md0
    out = {}
    # cumulative Z relative to the anchor, in TVT feet, /100 so it is O(1)
    out["zcum"] = -(z - z0) * 0.01
    # the per-step vertical rate dz/dmd (dimensionless), x10
    out["zrate"] = np.divide(dz, dmd, out=np.zeros(n),
                             where=np.abs(dmd) > 1e-9) * 10.0
    # (Z_anchor - Z)/(MD - MD_anchor): the LGBM `dz_rate`, its #1 gain feature
    out["zdisp_rate"] = np.clip(
        np.divide(z0 - z, dm, out=np.zeros(n), where=np.abs(dm) > 1.0), -1.0, 1.0) * 10.0
    # Z detrended in MD over the well -- the docs/27 `zperp`, the ONE Z form that
    # actually shipped (-0.049 ft, 5/5 folds, permutation null -0.0005).  The
    # measured-worse form pred + c*(Z - Z_anchor) is `zcum`, kept for contrast.
    A = np.stack([dm, np.ones(n)], 1)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    out["zperp"] = -(z - A @ coef) * 0.05
    miss = [c for c in names if c not in out]
    if miss:
        raise KeyError("unknown Z channel(s): %s (known: %s)" % (miss, Z_CHANNELS))
    return np.stack([out[c] for c in names], 1).astype(np.float32)


# =========================================================================== #
# M3 -- gre2 evidence channels, computed NATIVELY from the well + its typewell #
# =========================================================================== #
# Named exactly `gre2_<spec>_w<window>_<kind>`, where <spec>/<window>/<kind> are
# the screen's own identifiers (tools/gre2_screen.py, data/derived/_gre2/
# screen.csv), so a channel name here IS a row of the docs/37 table and the two
# cannot drift apart silently.  The spec objects themselves are taken from
# `rogii.gre2.SPECS` -- NOT re-declared -- so the reference / grid / emission /
# aggregator of every channel is the screened one by construction.
GRE2_PREFIX = "gre2_"
#: measured cost of the native path, accumulated by `gre2_channel_block` itself
#: so the number a run reports is the one it actually paid (cache hits included).
GRE2_STATS = {"secs": 0.0, "wells": 0, "rows": 0, "hits": 0}
_GRE2_RE = re.compile(r"^gre2_(?P<spec>[A-Za-z]+_[A-Za-z]+)_w(?P<win>\d+)_"
                      r"(?P<kind>[a-z]+)$")
GRE2_BASE_SPECS = {s.name: s for s in _gre2.SPECS}

# docs/37 Sec.5: the pseudo-typewell blends the well's own heel into the
# reference, so `shuffled_typewell` cannot destroy it and the mandatory control
# is INVALID for this family (F_pseudolog_w0_mean: partial 0.206 < its own
# partial-shuffle 0.228).  Refused by name, with the reason, in audit_channels.
GRE2_BANNED_SPECS = {
    "D_pseudo": "pseudo-typewell: the typewell-shuffle null is CONTAMINATED "
                "(docs/37 Sec.5) -- the control cannot destroy the heel blend",
    "F_pseudolog": "pseudo-typewell: null CONTAMINATED, partial 0.206 < its own "
                   "partial-shuffle 0.228 (docs/37 Sec.5)",
}
# docs/37 Sec.5 W4: 0 of the 144 non-`_mean` channels passes the BASE gate on
# either seed, and `_ent` / `_gap` sit UNDER their own null.
GRE2_ALLOWED_KINDS = ("mean",)
# docs/37 Sec.3: dominated everywhere by w0 or w500, on both seeds.
GRE2_BANNED_WINDOWS = (200,)

#: THE SELECTED gre2 SETS.  Every member is a row of the docs/37 Sec.3 table,
#: quoted as `far3 s17 / far3 s23  (delta vs BASE, folds won)`; BASE is the
#: incumbent `A_incumbent_w0_mean` == `gre_off_mean`, 0.3605 / 0.2453 on the two
#: seeds -- and docs/37 Sec.2 warns that the ABSOLUTE far3 of one 200-well
#: sample is NOT portable to another (the same incumbent reads 0.3605 / 0.2453 /
#: 0.3239 on three samples), so only the intra-sample MARGIN is readable.
GRE2_SETS = {
    "none": (),
    # the minimal arm: the single clean winner, alone
    "gre2e": ("gre2_E_callog_w0_mean",),
    # docs/37 Sec.6's own recommendation: E + the one second-rank channel whose
    # PARTIAL correlation stays clean on both seeds
    "gre2ec": ("gre2_E_callog_w0_mean", "gre2_C_cal_w1500_mean"),
    # THE SET.  E + every B_fine/C_cal survivor of the two gates on both seeds.
    "gre2a": (
        # -- the winner: typewell affine-calibrated on the heel, 2 ft grid, heel
        #    MAD emission (median removed), mean of LOG-LIKELIHOOD, cumulative
        #    from the PS.  far3 .4214 / .3165  (+.0609 / +.0712 vs BASE, 5/5 then
        #    4/5 folds; null .1939 / .1310).  Partial |corr| with u AFTER the
        #    incumbent is removed: .245 / .206 vs a partial-shuffle of .118 /
        #    .036 => r(E, incumbent) = .75, so 43 % of its variance is NEW.
        #    Decile profile .239 .281 .397 .409 .364 .344 .435 .461 .439 .365 vs
        #    the incumbent's .279 .291 .334 .316 .314 .362 .383 .382 .373 .326:
        #    it starts LOWER at the heel and ends HIGHER far out, i.e. it
        #    AMPLIFIES the F5 property instead of diluting it.
        "gre2_E_callog_w0_mean",
        # -- second rank, same calibrated reference, DENSITY aggregator: a real
        #    second point of view rather than a tuning of E.  Partial stays clean
        #    on both seeds (.144 / .169 vs .082 / .008).
        "gre2_C_cal_w1500_mean",   # far3 .3767 / .2902  (+.0162 / +.0449, 4/5 then 5/5)
        "gre2_C_cal_w500_mean",    # far3 .3764 / .2929  (+.0159 / +.0476, 3/5 then 5/5)
        # -- second rank, RAW typewell on the fine 2 ft grid.  CAVEAT, kept
        #    visible: on seed 23 B_fine's partial (.107) falls BELOW its own
        #    partial-shuffle (.125) => "the fine grid alone is a prettier version
        #    of the same channel, not a new one" (docs/37 Sec.4).  It clears both
        #    mandatory gates on both seeds, so it is in the set; it is the first
        #    thing to drop if the A/B says the set is too wide.
        "gre2_B_fine_w1500_mean",  # far3 .3962 / .2658  (+.0357 / +.0206, 4/5 and 4/5)
        "gre2_B_fine_w500_mean",   # far3 .4021 / .2656  (+.0416 / +.0203, 4/5 and 4/5)
    ),
}
# MEASURED HERE (10 wells, 52 529 native toe rows, tools/seqg_selftest.py's own
# sampling): the pooled cross-correlation of the 5 members of `gre2a` is
#     E    1.000  .624  .542  .554  .502
#     C15   .624 1.000  .865  .937  .847
#     C5    .542  .865 1.000  .828  .947
#     B15   .554  .937  .828 1.000  .915
#     B5    .502  .847  .947  .915 1.000
# => `E_callog_w0` really is a different object (r .50-.62 with the rest, the
# same picture docs/37 reports against the incumbent at r = .75), while the four
# DENSITY channels sit at r .83-.95 with each other -- they are four views of one
# thing.  `gre2ec` (E + C_cal_w1500) is the honest lean arm; `gre2a` spends 4
# input dimensions on that one thing and is the arm to shrink first if the A/B
# says the set is too wide.  This is a redundancy measurement, not a verdict:
# docs/29 F7 forbids reading it as evidence either way.
# NOT wired, each with its measurement:
#  * `gre2_F_pseudolog_w0_mean` -- far3 .3935 / .3196 would rank it 2nd, and it
#    is REFUSED: null contaminated (docs/37 Sec.5).  `GRE2_BANNED_SPECS` raises.
#  * `gre2_D_pseudo_w0_mean` -- same family, same refusal.
#  * `gre2_A_incumbent_w*_mean` -- the incumbent already reaches the net through
#    `gre_off_mean` in FEAT_SETS; its windowed variants add +.0099 at best.
#  * anything `_arg` / `_ent` / `_peak` / `_gap` / `_gapoff` / `_drift`, and
#    anything `_w200` -- refused by `audit_channels`, see the constants above.

# The gre2 sets are ALSO exposed through --feat-set, so `--feat-set gre2a` works
# and reads naturally; tools/train_seqwell.py splits the resolved list by prefix
# and routes the `gre2_` members to the native path instead of to the parquet.
FEAT_SETS.update({k: v for k, v in GRE2_SETS.items() if k != "none"})
#: combined arms.  docs/37 Sec.6 point 3: KEEP `gre_off_mean` -- gre2 extends it
#: (r = 0.75), it does not replace it.  These are the arms actually worth an A/B.
FEAT_SETS["gre1_gre2a"] = FEAT_SETS["gre1"] + GRE2_SETS["gre2a"]
FEAT_SETS["far15_gre2a"] = FEAT_SETS["far15"] + GRE2_SETS["gre2a"]


def gre2_parse(name: str):
    """`gre2_E_callog_w0_mean` -> ("E_callog", 0, "mean").  Raises otherwise."""
    m = _GRE2_RE.match(str(name))
    if not m:
        raise ValueError(
            "%r is not a gre2 channel name; expected "
            "gre2_<spec>_w<window>_<kind> with spec in %s"
            % (name, sorted(GRE2_BASE_SPECS)))
    return m.group("spec"), int(m.group("win")), m.group("kind")


def is_gre2(name: str) -> bool:
    return str(name).startswith(GRE2_PREFIX)


def split_feat_cols(cols) -> tuple[tuple, tuple]:
    """(parquet columns, gre2 columns), order preserved inside each group."""
    cols = tuple(cols)
    return (tuple(c for c in cols if not is_gre2(c)),
            tuple(c for c in cols if is_gre2(c)))


def audit_gre2(names, verbose=True) -> list:
    """Refuse, BY NAME AND WITH ITS MEASUREMENT, every family docs/37 killed."""
    bad = []
    for c in names:
        try:
            spec, win, kind = gre2_parse(c)
        except ValueError as e:
            bad.append(str(e))
            continue
        if spec not in GRE2_BASE_SPECS:
            bad.append("%s: unknown gre2 spec %r (known: %s)"
                       % (c, spec, sorted(GRE2_BASE_SPECS)))
        if spec in GRE2_BANNED_SPECS:
            bad.append("%s: %s" % (c, GRE2_BANNED_SPECS[spec]))
        if kind not in GRE2_ALLOWED_KINDS:
            bad.append("%s: kind %r refused -- docs/37 Sec.5 W4 measured 0 of the "
                       "144 non-`_mean` channels passing the BASE gate, and "
                       "`_ent`/`_gap` UNDER their own null (grid geometry, not "
                       "stratigraphy)" % (c, kind))
        if win in GRE2_BANNED_WINDOWS:
            bad.append("%s: window %d refused -- dominated by w0 or w500 on both "
                       "seeds (docs/37 Sec.3)" % (c, win))
    return bad


def gre2_specs_for(names) -> list:
    """The minimal list of Gre2Spec objects covering `names` (windows merged)."""
    want = {}
    for c in names:
        spec, win, _kind = gre2_parse(c)
        want.setdefault(spec, set()).add(win)
    return [_dc_replace(GRE2_BASE_SPECS[s], windows=tuple(sorted(w)))
            for s, w in sorted(want.items())]


def _gre2_cache_key(w, names, tw) -> str:
    """CONTENT hash.  Never the well name: the shuffle arm is the SAME well
    against a DIFFERENT typewell, and a name-keyed cache would hand the control
    the real well's evidence back -- exactly what the control exists to stop."""
    tw_t, tw_g = (w.tw_tvt, w.tw_gr) if tw is None else tw
    h = hashlib.sha1()
    for a in (np.asarray(w.md, np.float64), np.asarray(w.gr, np.float64),
              np.asarray(w.gr_obs, bool), np.asarray(w.tvt_in, np.float64),
              np.asarray(tw_t, np.float64), np.asarray(tw_g, np.float64)):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(("|%d|%d|%s" % (int(w.fe), int(w.n), ",".join(names))).encode())
    return h.hexdigest()


def gre2_channel_block(w, names, tw=None, cache_dir=None) -> np.ndarray:
    """The gre2 evidence channels for well `w`, on the model's own row grid.

    Reads `md, z, gr, gr_obs, tvt_in[:fe], fe` and the typewell `(tvt, gr)` --
    and NOTHING else.  `w.tvt`, `w.tops`, `w.tw_geol`, `w.datum` are never
    touched (rogii/gre2.py header; asserted at runtime by the smoke's
    gre2-truth-independence check and by tools/seqg_selftest.py T7).

    The evidence is accumulated along `np.arange(fe, n)` -- the NATIVE 1 ft toe
    rows, not feat_v6's 1-in-4 grid.  Heel rows are EDGE-CLAMPED to the first toe
    value, the same convention `align_feats` uses for the parquet channels: a
    legal test-safe quantity and no fabricated variation.

    `tw` overrides the typewell (the shuffle control passes the donor's), so the
    control re-computes the evidence against the donor instead of inheriting the
    real one.  It is zeroed on top of that -- see `TW_DERIVED_PREFIXES`.
    """
    names = list(names)
    n, fe = int(w.n), int(w.fe)
    out = np.zeros((n, len(names)), np.float32)
    if not names:
        return out
    _t0 = time.time()
    GRE2_STATS["wells"] += 1
    GRE2_STATS["rows"] += n
    cpath = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cpath = os.path.join(cache_dir, _gre2_cache_key(w, names, tw) + ".npy")
        if os.path.exists(cpath):
            cached = np.load(cpath)
            if cached.shape == out.shape:
                GRE2_STATS["hits"] += 1
                GRE2_STATS["secs"] += time.time() - _t0
                return np.ascontiguousarray(cached, np.float32)
    rows = np.arange(fe, n, dtype=np.int64)
    if len(rows) == 0:
        GRE2_STATS["secs"] += time.time() - _t0
        return out
    vals = {}
    for spec in gre2_specs_for(names):
        vals.update(_gre2.channels(w, spec, rows=rows, tw=tw))
    for j, c in enumerate(names):
        v = np.nan_to_num(np.asarray(vals[c[len(GRE2_PREFIX):]], np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        if len(v) != len(rows):
            raise AssertionError("gre2 channel %s: %d values for %d rows"
                                 % (c, len(v), len(rows)))
        out[fe:, j] = v
        out[:fe, j] = v[0]                     # edge-clamp the heel
    if not np.all(np.isfinite(out)):
        raise AssertionError("non-finite gre2 channel in %s" % w.name)
    if cpath:
        np.save(cpath, out)
    GRE2_STATS["secs"] += time.time() - _t0
    return out


def audit_channels(feat_cols, z_cols, schema_names=None, gre2_cols=(),
                   verbose=True) -> dict:
    """LEAK AUDIT.  Runs before anything is built; raises on any violation."""
    rep = {"feat": list(feat_cols), "z": list(z_cols),
           "gre2": list(gre2_cols), "violations": []}
    for c in feat_cols:
        if c in BANNED_COLS:
            rep["violations"].append("%s: TRUTH / key column" % c)
        for pre in CROSS_WELL_PREFIXES:
            if c.startswith(pre):
                rep["violations"].append("%s: cross-well prefix %r" % (c, pre))
        if schema_names is not None and c not in schema_names:
            rep["violations"].append("%s: absent from the feature parquet" % c)
    for c in z_cols:
        if c not in Z_CHANNELS:
            rep["violations"].append("%s: unknown Z channel" % c)
    rep["violations"] += audit_gre2(gre2_cols, verbose=verbose)
    allc = list(feat_cols) + list(z_cols) + list(gre2_cols)
    rep["tw_derived"] = [c for c in allc
                         if any(c.startswith(p) for p in TW_DERIVED_PREFIXES)]
    missing_neutral = [c for c in gre2_cols if c not in rep["tw_derived"]]
    if missing_neutral:
        rep["violations"].append(
            "gre2 channel(s) NOT covered by TW_DERIVED_PREFIXES, so the "
            "typewell-shuffle control would keep the real GR evidence: %s"
            % missing_neutral)
    if verbose:
        print("[leak-audit] %d feat + %d Z + %d gre2 channels"
              % (len(feat_cols), len(z_cols), len(gre2_cols)))
        print("[leak-audit]   source            = %s (intra-well certified: "
              "schema guard + bit-parity, docs/README 4)" % os.path.basename(FEAT_PARQUET))
        print("[leak-audit]   truth columns     = NONE  (w.tvt/tops/tw_geol/datum "
              "never read; y_delta/y_tvt/_tvt_last banned by name)")
        print("[leak-audit]   cross-well prefix = NONE  (%d prefixes checked)"
              % len(CROSS_WELL_PREFIXES))
        print("[leak-audit]   Z channels        = %s  (from w.md/w.z/w.fe only)"
              % (list(z_cols) or "none"))
        print("[leak-audit]   gre2 channels     = %s"
              % (list(gre2_cols) or "none"))
        if gre2_cols:
            print("[leak-audit]     computed NATIVELY from md/z/gr/gr_obs/"
                  "tvt_in[:fe]/fe + the typewell (rogii/gre2.py); no parquet, "
                  "no cross-well term; %d refused families checked "
                  "(D_pseudo/F_pseudolog null-contaminated, non-`_mean`, _w200)"
                  % (len(GRE2_BANNED_SPECS) + 2))
        print("[leak-audit]   typewell-derived  = %d/%d -> neutralised in the "
              "SHUFFLE control%s"
              % (len(rep["tw_derived"]), len(allc),
                 "  (ALL %d gre2 included)" % len(gre2_cols) if gre2_cols else ""))
        print("[leak-audit]   VIOLATIONS        = %s"
              % (rep["violations"] or "none"))
    if rep["violations"]:
        raise AssertionError("channel leak audit FAILED: %s" % rep["violations"])
    return rep


class FeatSource:
    """feat_v6 columns, grouped per well, on their own 1-in-`sub` toe row grid."""

    def __init__(self, cols, path=None, wells=None, verbose=True):
        import pandas as pd
        import pyarrow.parquet as pq
        self.path = path or FEAT_PARQUET
        self.cols = list(cols)
        self.by = {}
        if not self.cols:
            self.schema = None
            return
        self.schema = list(pq.ParquetFile(self.path).schema_arrow.names)
        audit_channels(self.cols, (), self.schema, verbose=False)
        df = pd.read_parquet(self.path, columns=["well", "row"] + self.cols)
        if wells is not None:
            df = df[df["well"].isin(set(wells))]
        for name, g in df.groupby("well", sort=False):
            self.by[str(name)] = (g["row"].to_numpy(np.int64),
                                  g[self.cols].to_numpy(np.float64))
        if verbose:
            print("[featsrc] %s: %d cols x %d wells (%d rows kept)"
                  % (os.path.basename(self.path), len(self.cols), len(self.by), len(df)))

    def assert_varying(self, min_frac=MIN_FRAC_WITHIN, verbose=True):
        """docs/29 gate -- refuse any column that is a per-well constant."""
        if not self.cols:
            return {}
        k = len(self.cols)
        wi = np.zeros(k); tot_s = np.zeros(k); tot_ss = np.zeros(k); nrow = 0
        for _, (_rows, vals) in self.by.items():
            x = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            wi += x.var(0) * len(x)
            tot_s += x.sum(0); tot_ss += (x * x).sum(0); nrow += len(x)
        within = np.sqrt(wi / max(nrow, 1))
        total = np.sqrt(np.maximum(tot_ss / max(nrow, 1)
                                   - (tot_s / max(nrow, 1)) ** 2, 0.0))
        frac = np.where(total > 1e-12, within / np.maximum(total, 1e-12), 0.0)
        bad = [(c, float(f)) for c, f in zip(self.cols, frac) if f < min_frac]
        if verbose:
            print("[docs/29 gate] within-well sd / total sd, per channel: "
                  "min %.3f  median %.3f  (threshold %.3f)"
                  % (frac.min(), float(np.median(frac)), min_frac))
        if bad:
            raise AssertionError(
                "per-well CONSTANT channel(s) rejected (docs/29: aggregated "
                "per-well descriptors are worth exactly zero): %s" % bad)
        return dict(zip(self.cols, frac.tolist()))

    def get(self, name):
        return self.by.get(str(name))


def align_feats(rows, vals, n) -> tuple[np.ndarray, int]:
    """Map feat_v6's toe rows onto the model's `np.arange(n)` grid.

    `np.interp` on the ROW INDEX: exact at the sampled rows, linear between,
    edge-clamped outside (heel rows take the first sampled toe value).  NaN are
    dropped per column before the interpolation; an all-NaN column becomes 0.
    Returns (array [n,k], number of all-NaN columns).
    """
    x = np.arange(n, dtype=float)
    r = np.asarray(rows, float)
    out = np.zeros((n, vals.shape[1]), np.float64)
    dead = 0
    for j in range(vals.shape[1]):
        v = vals[:, j]
        ok = np.isfinite(v)
        if not ok.any():
            dead += 1
            continue
        out[:, j] = np.interp(x, r[ok], v[ok])
    if not np.all(np.isfinite(out)):
        raise AssertionError("NaN/inf leaked through align_feats")
    return out.astype(np.float32), dead


class ChanNorm:
    """Per-channel standardisation fitted on the TRAINING WELLS ONLY."""

    def __init__(self, names):
        self.names = list(names)
        self.mean = np.zeros(len(self.names), np.float32)
        self.std = np.ones(len(self.names), np.float32)
        self.fitted = False

    def fit(self, wells, train_idx):
        k = len(self.names)
        if k == 0:
            self.fitted = True
            return self
        s = np.zeros(k); ss = np.zeros(k); n = 0
        for i in train_idx:
            e = wells[int(i)].extra.astype(np.float64)
            s += e.sum(0); ss += (e * e).sum(0); n += len(e)
        m = s / max(n, 1)
        v = np.maximum(ss / max(n, 1) - m * m, 0.0)
        sd = np.sqrt(v)
        sd[sd < 1e-6] = 1.0
        self.mean = m.astype(np.float32)
        self.std = sd.astype(np.float32)
        self.fitted = True
        return self

    def apply(self, ws, neutral: np.ndarray | None = None):
        if not self.names:
            return
        if not self.fitted:
            raise AssertionError("ChanNorm.apply before fit -- that is the leak")
        ws.extra = (ws.extra - self.mean[None, :]) / self.std[None, :]
        if neutral is not None and neutral.any():
            ws.extra[:, neutral] = 0.0      # = the TRAIN mean, post-standardisation
        ws.extra = np.ascontiguousarray(ws.extra, np.float32)

    def to_dict(self):
        return dict(names=list(self.names), mean=self.mean.tolist(),
                    std=self.std.tolist(), fitted=bool(self.fitted))

    @classmethod
    def from_dict(cls, d):
        o = cls(d["names"])
        o.mean = np.asarray(d["mean"], np.float32)
        o.std = np.asarray(d["std"], np.float32)
        o.fitted = bool(d.get("fitted", True))
        return o


@dataclass
class SeqParams:
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.1
    res_scale: float = 0.05        # U6
    noise_sigma: float = 0.08      # spec
    noise_clip: float = 15.0       # spec
    delta_scale: float = 0.02      # U7
    abs_scale: float = 3.0         # U7
    window: int = 256              # spec
    warmup: int = 64               # U8
    gr_heel_affine: bool = False   # U4
    prior_beta: float = 0.0        # M1 -- spec value is 1.0, measured 105.6 vs 15.5
    feat_cols: tuple = ()          # M2 -- feat_v6 per-row channels, () = legacy 37-dim
    z_cols: tuple = ()             # M2 -- Z-derived per-row channels
    gre2_cols: tuple = ()          # M3 -- gre2 evidence channels, computed natively

    # channel order in `extra` is feat | z | gre2, and `chan_names` is the ONE
    # place that says so: ChanNorm, the checkpoint and every slice below read it.
    @property
    def n_extra(self) -> int:
        return len(self.feat_cols) + len(self.z_cols) + len(self.gre2_cols)

    @property
    def chan_names(self) -> list:
        return list(self.feat_cols) + list(self.z_cols) + list(self.gre2_cols)

    @property
    def z_slice(self) -> slice:
        return slice(len(self.feat_cols), len(self.feat_cols) + len(self.z_cols))

    @property
    def gre2_slice(self) -> slice:
        s = len(self.feat_cols) + len(self.z_cols)
        return slice(s, s + len(self.gre2_cols))

    def tw_derived_mask(self) -> np.ndarray:
        """True for channels computed against the typewell (shuffle control)."""
        return np.array([any(c.startswith(p) for p in TW_DERIVED_PREFIXES)
                         for c in self.chan_names], bool)

    def to_dict(self):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["feat_cols"] = list(d["feat_cols"])
        d["z_cols"] = list(d["z_cols"])
        d["gre2_cols"] = list(d["gre2_cols"])
        return d


# =========================================================================== #
# per-well container -- TEST SCHEMA ONLY                                      #
# =========================================================================== #
class WellSeq:
    """One complete well, packed for the sequence model.

    Built from MD/X/Y/Z/GR/TVT_input + typewell TVT/GR.  `r_true` (the training
    target) is the ONLY field that reads `w.tvt`, and it is None whenever the
    well has no truth.
    """
    __slots__ = ("name", "n", "fe", "base", "dz", "grn", "tvt_in", "anchor",
                 "known", "tw", "tw_t0", "tw_n", "dtvt_true", "tvt_true", "is_eval",
                 "extra", "feat_missing", "feat_dead")

    def __init__(self, w, tw=None, name: str | None = None,
                 p: SeqParams | None = None, feats=None, gre2_cache=None):
        p = p or SeqParams()
        self.name = name or w.name
        n = int(w.n)
        self.n = n
        self.fe = int(w.fe)

        tw_tvt, tw_gr = (np.asarray(w.tw_tvt, float), np.asarray(w.tw_gr, float)) \
            if tw is None else (np.asarray(tw[0], float), np.asarray(tw[1], float))

        # ---- GR on one shared scale (the TYPEWELL's) -----------------------
        mu = float(np.mean(tw_gr))
        sd = float(np.std(tw_gr))
        sd = sd if sd > 1e-6 else 1.0
        gr = np.asarray(w.gr, float)
        if p.gr_heel_affine and self.fe > 8:      # U4, OFF by default
            th = np.asarray(w.tvt_in[:self.fe], float)
            ok = np.isfinite(th) & np.asarray(w.gr_obs[:self.fe], bool)
            if ok.sum() > 32:
                ref = np.interp(th[ok], tw_tvt, tw_gr)
                A = np.stack([ref, np.ones(ok.sum())], 1)
                g, b = np.linalg.lstsq(A, gr[:self.fe][ok], rcond=None)[0]
                if abs(g) > 1e-3:
                    gr = (gr - b) / g
        grn = (gr - mu) / sd

        # ---- typewell resampled onto the uniform 0.25 ft grid ---------------
        t0 = float(np.min(tw_tvt)) - TW_PAD
        t1 = float(np.max(tw_tvt)) + TW_PAD
        G = int(math.ceil((t1 - t0) / TW_DT)) + 1
        grid = t0 + np.arange(G) * TW_DT
        twn = (np.interp(grid, tw_tvt, tw_gr) - mu) / sd   # np.interp edge-clamps
        self.tw = twn.astype(np.float32)
        self.tw_t0 = float(t0)
        self.tw_n = G

        # ---- base features --------------------------------------------------
        def d(a):
            a = np.asarray(a, float)
            o = np.zeros(n)
            o[1:] = np.diff(a)
            return o

        b = np.zeros((n, N_BASE), np.float32)
        b[:, 0] = d(w.md)
        b[:, 1] = d(w.x)
        b[:, 2] = d(w.y)
        b[:, 3] = d(w.z)
        b[:, 4] = grn
        b[:, 5] = d(grn)
        b[:, 6] = np.asarray(w.gr_obs, float)
        self.base = (b * FEAT_SCALE[None, :]).astype(np.float32)
        self.dz = d(w.z).astype(np.float32)
        self.grn = grn.astype(np.float32)

        ti = np.asarray(w.tvt_in, float)
        self.tvt_in = ti.astype(np.float32)
        self.known = np.isfinite(ti)
        self.is_eval = ~self.known
        self.anchor = float(ti[self.fe - 1]) if self.fe > 0 else float(np.nanmedian(ti))

        # ---- TARGET (train only; never an input) ----------------------------
        # The residual target is assembled at loss time as
        #     r_true = dTVT + prior_beta * dZ
        # so that prior_beta stays a runtime knob (M1) instead of a baked-in axis.
        tv = getattr(w, "tvt", None)
        if tv is not None and np.all(np.isfinite(tv)):
            tv = np.asarray(tv, float)
            self.tvt_true = tv.astype(np.float32)
            self.dtvt_true = d(tv).astype(np.float32)
        else:
            self.tvt_true = None
            self.dtvt_true = None

        # ---- M2: extra per-row channels (RAW; ChanNorm standardises later) ---
        self.feat_missing = False
        self.feat_dead = 0
        k = p.n_extra
        if k == 0:
            self.extra = None
        else:
            blocks = []
            if p.feat_cols:
                fv = feats.get(self.name) if feats is not None else None
                if fv is None:
                    self.feat_missing = True
                    blocks.append(np.zeros((n, len(p.feat_cols)), np.float32))
                else:
                    a, dead = align_feats(fv[0], fv[1], n)
                    self.feat_dead = dead
                    blocks.append(a)
            if p.z_cols:
                blocks.append(z_channel_block(w.md, w.z, self.fe, p.z_cols))
            if p.gre2_cols:
                # M3.  `tw` is the SHUFFLE control's donor typewell when the
                # control builds this well, so the evidence is re-derived
                # against the donor rather than inherited from the real one.
                blocks.append(gre2_channel_block(
                    w, p.gre2_cols,
                    tw=(None if tw is None else (tw_tvt, tw_gr)),
                    cache_dir=gre2_cache))
            self.extra = np.ascontiguousarray(
                np.concatenate(blocks, 1) if len(blocks) > 1 else blocks[0],
                np.float32)
            if self.extra.shape != (n, k):
                raise AssertionError("extra shape %s != (%d,%d)"
                                     % (self.extra.shape, n, k))
            if not np.all(np.isfinite(self.extra)):
                raise AssertionError("non-finite extra channel in %s" % self.name)


def build_wellseq(w, tw=None, p: SeqParams | None = None, feats=None,
                  gre2_cache=None) -> WellSeq:
    return WellSeq(w, tw=tw, p=p, feats=feats, gre2_cache=gre2_cache)


# =========================================================================== #
# GPU-side well bank + differentiable typewell lookup                         #
# =========================================================================== #
class TwBank:
    """All typewell grids in one flat tensor; lookup by well id, no per-batch copy."""

    def __init__(self, wells: list[WellSeq], device):
        self.G = max(ws.tw_n for ws in wells)
        flat = np.zeros((len(wells), self.G), np.float32)
        for i, ws in enumerate(wells):
            flat[i, :ws.tw_n] = ws.tw
            flat[i, ws.tw_n:] = ws.tw[-1]          # edge-clamp the pad
        self.flat = torch.from_numpy(flat.reshape(-1)).to(device)
        self.t0 = torch.tensor([ws.tw_t0 for ws in wells], dtype=torch.float32,
                               device=device)
        self.nmax = torch.tensor([ws.tw_n - 1 for ws in wells], dtype=torch.float32,
                                 device=device)
        self.device = device

    def lookup(self, wid: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """wid [B] long ; t [B, ...] float -> typewell GR_n at those TVTs.

        Linear interpolation on the 0.25 ft grid via two flat gathers.  The
        fractional weight carries the gradient d(out)/d(t) -- this is the
        differentiable `torch.gather` lookup the spec requires for phase 2.
        """
        shp = t.shape
        B = shp[0]
        tf = t.reshape(B, -1)
        idx = (tf - self.t0[wid][:, None]) / TW_DT
        idx = torch.clamp(idx, torch.zeros_like(idx), self.nmax[wid][:, None] - 1e-3)
        i0 = torch.floor(idx)
        fr = idx - i0
        i0l = i0.long()
        off = (wid.long() * self.G)[:, None]
        g0 = self.flat[off + i0l]
        g1 = self.flat[off + i0l + 1]
        return (g0 + (g1 - g0) * fr).reshape(shp)

    def profile(self, wid: torch.Tensor, t: torch.Tensor,
                grn: torch.Tensor) -> torch.Tensor:
        """t [B,T] (or [B]) running TVT, grn [B,T] observed GR_n -> [B,T,29].

        prof[..., j] = tw_gr_n(t + OFFSETS[j]) - gr_n(here).  Where the vector
        crosses zero is where the observed log would match the typewell.
        """
        offs = _OFF_T[t.device]
        tt = t.unsqueeze(-1) + offs                       # [B,T,29]
        v = self.lookup(wid, tt)
        return v - grn.unsqueeze(-1)


class _OffCache(dict):
    def __missing__(self, dev):
        v = torch.from_numpy(OFFSETS).to(dev)
        self[dev] = v
        return v


_OFF_T = _OffCache()


# =========================================================================== #
# batching                                                                    #
# =========================================================================== #
def pack_windows(wells: list[WellSeq], wid: np.ndarray, start: np.ndarray,
                 L: int, device, need_target: bool = True) -> dict:
    """B windows of length L, window b taken from wells[wid[b]] at [start[b], +L)."""
    B = len(wid)
    base = np.zeros((B, L, N_BASE), np.float32)
    dz = np.zeros((B, L), np.float32)
    grn = np.zeros((B, L), np.float32)
    tvt = np.zeros((B, L), np.float32)
    rt = np.zeros((B, L), np.float32)
    msk = np.zeros((B, L), np.float32)
    K = 0 if wells[wid[0]].extra is None else wells[wid[0]].extra.shape[1]
    ext = np.zeros((B, L, K), np.float32) if K else None
    for b in range(B):
        ws = wells[wid[b]]
        s = int(start[b]); e = min(s + L, ws.n); m = e - s
        base[b, :m] = ws.base[s:e]
        dz[b, :m] = ws.dz[s:e]
        grn[b, :m] = ws.grn[s:e]
        msk[b, :m] = 1.0
        if K:
            ext[b, :m] = ws.extra[s:e]
        if need_target and ws.tvt_true is not None:
            tvt[b, :m] = ws.tvt_true[s:e]
            rt[b, :m] = ws.dtvt_true[s:e]
    t = lambda a: torch.from_numpy(a).to(device)          # noqa: E731
    return dict(base=t(base), dz=t(dz), grn=t(grn), tvt=t(tvt), dtvt=t(rt),
                mask=t(msk), extra=(t(ext) if K else None),
                wid=torch.from_numpy(wid.astype(np.int64)).to(device))


def pack_full(wells: list[WellSeq], wid: np.ndarray, device) -> dict:
    """Whole wells, right-padded, for full-well rollout evaluation."""
    B = len(wid)
    T = max(wells[i].n for i in wid)
    base = np.zeros((B, T, N_BASE), np.float32)
    dz = np.zeros((B, T), np.float32)
    grn = np.zeros((B, T), np.float32)
    tin = np.zeros((B, T), np.float32)
    kn = np.zeros((B, T), np.float32)
    tvt = np.zeros((B, T), np.float32)
    ev = np.zeros((B, T), np.float32)
    fe = np.zeros(B, np.int64)
    anc = np.zeros(B, np.float32)
    K = 0 if wells[wid[0]].extra is None else wells[wid[0]].extra.shape[1]
    ext = np.zeros((B, T, K), np.float32) if K else None
    for b, i in enumerate(wid):
        ws = wells[i]
        n = ws.n
        base[b, :n] = ws.base
        dz[b, :n] = ws.dz
        grn[b, :n] = ws.grn
        kn[b, :n] = ws.known
        tin[b, :n] = np.where(ws.known, np.nan_to_num(ws.tvt_in), 0.0)
        ev[b, :n] = ws.is_eval
        fe[b] = ws.fe
        anc[b] = ws.anchor
        if K:
            ext[b, :n] = ws.extra
            ext[b, n:] = ws.extra[-1]          # edge-clamp the right pad
        if ws.tvt_true is not None:
            tvt[b, :n] = ws.tvt_true
    t = lambda a: torch.from_numpy(a).to(device)          # noqa: E731
    return dict(base=t(base), dz=t(dz), grn=t(grn), tvt_in=t(tin), known=t(kn),
                tvt=t(tvt), is_eval=t(ev), fe=fe, anchor=t(anc), T=T,
                extra=(t(ext) if K else None),
                wid=torch.from_numpy(np.asarray(wid, np.int64)).to(device))


def drift_noise(shape, sigma: float, clip: float, device, gen=None) -> torch.Tensor:
    """Clipped cumulative random walk -- the spec's drift-noise corruption."""
    e = torch.randn(shape, device=device, generator=gen) * sigma
    w = torch.cumsum(e, dim=1)
    return torch.clamp(w, -clip, clip)


# =========================================================================== #
# the model                                                                   #
# =========================================================================== #
class SeqWellNet(nn.Module):
    """M2: the extra per-row channels enter in TWO places, both at the index of
    the step being predicted:

      * the GRU input at index i  -- so the recurrent state carries their history;
      * the HEAD at index i       -- alongside `prof_next`, so the residual r_i has
        a DIRECT path to row i's evidence, exactly as it already has to row i's
        typewell profile.  Without this the channels would have to survive the
        one-step delay of the recurrence to reach the output.

    With `n_extra == 0` both concatenations are no-ops and the module is
    bit-identical to the 37-dim legacy net (verified by --feat-set none).
    """

    def __init__(self, p: SeqParams | None = None):
        super().__init__()
        self.p = p or SeqParams()
        self.n_extra = int(self.p.n_extra)
        self.in_dim = N_BASE + 1 + NPROF + self.n_extra    # 7 + d_TVT + 29 [+ K]
        self.head_dim = self.p.hidden + NPROF + self.n_extra
        self.ln = nn.LayerNorm(self.in_dim)       # LayerNorm, NEVER BatchNorm
        self.gru = nn.GRU(self.in_dim, self.p.hidden, self.p.layers,
                          batch_first=True,
                          dropout=self.p.dropout if self.p.layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(self.head_dim, self.p.hidden),
            nn.ReLU(),
            nn.Dropout(self.p.dropout),
            nn.Linear(self.p.hidden, 1),
        )
        # U6: start exactly on the structural prior TVT = -Z + const
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    # ---- helpers ----------------------------------------------------------
    def _x(self, base, dtvt, prof, extra=None):
        parts = [base, (dtvt * DTVT_SCALE).unsqueeze(-1), prof]
        if self.n_extra:
            if extra is None:
                raise AssertionError(
                    "SeqWellNet built with %d extra channels but the batch "
                    "carries none -- the WellSeq was built with a different "
                    "SeqParams" % self.n_extra)
            parts.append(extra)
        return torch.cat(parts, dim=-1)

    def _r(self, htop, prof_next, extra_next=None):
        parts = [htop, prof_next]
        if self.n_extra:
            if extra_next is None:
                raise AssertionError("head expects %d extra channels, got none"
                                     % self.n_extra)
            parts.append(extra_next)
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1) * self.p.res_scale

    @staticmethod
    def _sl(extra, sl):
        """Slice the [B,T,K] extra block, tolerating None (legacy path)."""
        return None if extra is None else extra[:, sl]

    # ---- teacher forcing over a whole window ------------------------------
    def forward_tf(self, bank: TwBank, batch: dict, track: torch.Tensor):
        """track [B,L] = the (possibly noisy) TVT history fed to the net.

        Returns r_hat [B,L-1] and t_pred [B,L-1]: the one-step-ahead absolute
        prediction for steps 1..L-1, made from `track` and the profiles.
        """
        wid = batch["wid"]
        ex = batch.get("extra")
        dtvt = torch.zeros_like(track)
        dtvt[:, 1:] = track[:, 1:] - track[:, :-1]
        prof = bank.profile(wid, track, batch["grn"])
        hout, _ = self.gru(self.ln(self._x(batch["base"], dtvt, prof, ex)))
        # "the NEXT step's typewell profile", taken at the zero-residual
        # structural prediction for that step (M1: weighted by prior_beta).
        t_base = track[:, :-1] - self.p.prior_beta * batch["dz"][:, 1:]
        prof_next = bank.profile(wid, t_base, batch["grn"][:, 1:])
        # M2: the head of step j sees row j's channels -- the same index as
        # prof_next and as grn[:, 1:].
        r_hat = self._r(hout[:, :-1], prof_next, self._sl(ex, slice(1, None)))
        return r_hat, t_base + r_hat

    # ---- one autoregressive step ------------------------------------------
    def step(self, bank: TwBank, wid, base_i, dz_i, grn_i, t_cur, h, htop,
             extra_i=None, snap_val=None, snap_mask=None):
        t_base = t_cur - self.p.prior_beta * dz_i
        prof_next = bank.profile(wid, t_base.unsqueeze(1), grn_i.unsqueeze(1))[:, 0]
        r = self._r(htop, prof_next, extra_i)
        t_new = t_base + r
        if snap_mask is not None:
            # inference only: wherever TVT_input is known, snap BEFORE the state
            # update so the GRU never carries a position we know to be wrong.
            t_new = torch.where(snap_mask, snap_val, t_new)
        prof_cur = bank.profile(wid, t_new.unsqueeze(1), grn_i.unsqueeze(1))[:, 0]
        # every argument here is [B, ...]: cat on the last axis, then add the
        # length-1 time axis the GRU wants.
        x = self._x(base_i, (t_new - t_cur), prof_cur, extra_i).unsqueeze(1)
        hout, h = self.gru(self.ln(x), h)
        return t_new, r, h, hout[:, 0]

    # ---- K-step rollout inside a packed window -----------------------------
    def rollout_window(self, bank: TwBank, batch: dict, t_start, h, htop,
                       s: int, K: int):
        """Roll K steps starting at index s (predicting s .. s+K-1)."""
        wid = batch["wid"]
        ex = batch.get("extra")
        ts, rs = [], []
        t_cur = t_start
        for k in range(K):
            i = s + k
            t_cur, r, h, htop = self.step(
                bank, wid, batch["base"][:, i], batch["dz"][:, i],
                batch["grn"][:, i], t_cur, h, htop,
                extra_i=self._sl(ex, i))
            ts.append(t_cur); rs.append(r)
        return torch.stack(ts, 1), torch.stack(rs, 1), h, htop

    # ---- warm-up (teacher forced on a given track) -------------------------
    def warmup(self, bank: TwBank, batch: dict, track: torch.Tensor, s0: int, s1: int):
        """Teacher-force over [s0, s1); return (h, htop) at index s1-1."""
        wid = batch["wid"]
        seg = slice(s0, s1)
        tr = track[:, seg]
        dtvt = torch.zeros_like(tr)
        dtvt[:, 1:] = tr[:, 1:] - tr[:, :-1]
        prof = bank.profile(wid, tr, batch["grn"][:, seg])
        hout, h = self.gru(self.ln(self._x(batch["base"][:, seg], dtvt, prof,
                                           self._sl(batch.get("extra"), seg))))
        return h, hout[:, -1]

    # ---- full-well inference ------------------------------------------------
    @torch.no_grad()
    def predict_full(self, bank: TwBank, fb: dict, chunk_log=None):
        """Encode the known prefix anchored on TVT_input, then step to TD.

        Wherever TVT_input is known the running value SNAPS BACK to it.
        Returns pred [B,T] (float32, the model's TVT everywhere).
        """
        self.eval()
        wid = fb["wid"]
        B, T = fb["base"].shape[0], fb["T"]
        dev = fb["base"].device
        fe_min = int(min(fb["fe"]))
        # prefix: teacher-force on the KNOWN TVT_input (this is the anchor path)
        track = fb["tvt_in"].clone()
        s1 = max(fe_min, 2)
        h, htop = self.warmup(bank, fb, track, 0, s1)
        pred = torch.zeros(B, T, device=dev)
        pred[:, :s1] = track[:, :s1]
        t_cur = track[:, s1 - 1]
        kn_all = fb["known"] > 0.5
        any_known = kn_all[:, s1:].any(dim=0)
        ex = fb.get("extra")
        for i in range(s1, T):
            km = kn_all[:, i] if bool(any_known[i - s1]) else None
            t_cur, r, h, htop = self.step(
                bank, wid, fb["base"][:, i], fb["dz"][:, i], fb["grn"][:, i],
                t_cur, h, htop,
                extra_i=self._sl(ex, i),
                snap_val=(fb["tvt_in"][:, i] if km is not None else None),
                snap_mask=km)
            pred[:, i] = t_cur
        return pred


# =========================================================================== #
# losses                                                                      #
# =========================================================================== #
def blend_loss(r_hat, r_true, t_hat, t_true, mask, p: SeqParams):
    """50/50 blend of delta-RMSE and absolute-TVT RMSE, each unit-normalised (U7)."""
    m = mask
    den = m.sum().clamp(min=1.0)
    d = torch.sqrt((((r_hat - r_true) ** 2) * m).sum() / den + 1e-12)
    a = torch.sqrt((((t_hat - t_true) ** 2) * m).sum() / den + 1e-12)
    return 0.5 * d / p.delta_scale + 0.5 * a / p.abs_scale, float(d), float(a)


def pooled_rmse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray):
    m = mask.astype(bool)
    return float(np.sqrt(np.mean((pred[m] - true[m]) ** 2)))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
