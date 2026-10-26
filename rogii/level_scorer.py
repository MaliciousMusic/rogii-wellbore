"""rogii.level_scorer — STEP 2 du levier de NIVEAU: le SCORER LISTWISE.

CONTEXTE (docs/21 §12.3, source S15 myckeluribe — le SEUL levier de niveau
positif et deployable de tout le corpus)
-------------------------------------------------------------------------
Le niveau n'est PAS un probleme de regression continue: c'est un probleme de
SELECTION DISCRETE. Le cout d'appariement GR sur tout le lateral,
    J(D) = cout(chemin TVT predit + D  vs  typewell),  D dans [-60, +60] ft,
est multimodal; ses minima locaux (~14 par puits) CONTIENNENT la verite a
+-5 ft dans 97 % des puits. Un petit CNN 1-D score les candidats LISTWISE, et
la LECTURE EST LA MOYENNE A POSTERIORI sur les candidats, jamais l'argmax.

Mesure de la source: pooled 8.9134 -> 8.6986, bootstrap par blocs de puits
apparie = 1.000; puits catastrophiques top-1 0 % -> 21 %. Le gain CROIT quand
la base s'affaiblit (-0.457 sur une base a 11.19). Son unique echec (+0.80)
etait sur une base a 7.21, diagnostique arithmetiquement: les mouvements du
correcteur (~3.5 ft) depassaient le residu de cette base,
sqrt(7.21^2 + 3.5^2) = 8.01. NOTRE BASE EST 11.69 -> dans la bande visee.

Ce module ne contient QUE le scorer (modele + dataset + metriques). Il ne
genere PAS les candidats: il consomme un "banc de candidats" ecrit par
l'enumerateur (contrat NPZ ci-dessous). Tant que l'enumerateur n'a pas atterri,
`tools/train_level_scorer.py --source synth` fabrique un banc synthetique qui a
la MEME forme, ce qui prouve toute la plomberie.

REGLE D'OR (identique au reste du depot): INTRA-PUITS STRICT. Aucune colonne
train-only sur un chemin d'inference. `e_sum` / `e_sq` (dont on derive la
verite de niveau) sont utilises UNIQUEMENT pour construire la cible
d'entrainement sur les folds d'ENTRAINEMENT et pour EVALUER hors-fold.

CONTRAT NPZ DU BANC DE CANDIDATS  (ce que l'enumerateur doit ecrire)
-------------------------------------------------------------------
Ragged concat + pointeurs. W puits, Ktot candidats, L points de sequence.

  wells     (W,)   <U8    nom du puits
  fold      (W,)   int64  fold canonique (data/derived/folds.csv)
  n_row     (W,)   int64  nombre de lignes de TOE (le denominateur poole)
  e_sum     (W,)   f8     somme sur le toe de (y_tvt - pred_base)   [TRAIN]
  e_sq      (W,)   f8     somme sur le toe de (y_tvt - pred_base)^2 [TRAIN]
  ptr       (W+1,) int64  bornes des candidats du puits i: ptr[i]:ptr[i+1]
  off       (Ktot,) f4    offset D_k du candidat, en ft
  cost      (Ktot,) f4    J(D_k), le cout d'appariement GR au candidat
  gr        (W, L) f4     GR MESURE le long du toe, reechantillonne
  obs       (W, L) f4     fraction de GR reellement observe dans le bloc
  shape     (W, L) f4     chemin de base (S = TVT+Z) relatif a l'ancre, en ft
  tw        (Ktot, L, 3) f4  typewell lu aux profondeurs decalees du candidat,
                             canaux = (-3 ft, 0 ft, +3 ft) de contexte
  obs_frac  (W,)   f4     fraction de GR observe sur le puits
  toe_len   (W,)   f4     longueur du toe en ft

VERITE DE NIVEAU: D* = e_sum / n_row  (le meilleur offset constant du puits;
c'est exactement l'oracle O1 de docs/23). Aucune autre colonne de verite n'est
lue. Elle est NaN cote test, et le code refuse alors d'entrainer sur ce puits.

CHOIX QUI SONT LES MIENS (le corpus ne les specifie pas) — marques [MOI]:
  [MOI] L = 512 points de sequence, par DECIMATION EN MOYENNE DE BLOC (pas
        d'interp naive: un toe de ~4 700 lignes ramene a 512 par interpolation
        aliase les ondulations GR; la moyenne de bloc est l'anti-aliasing juste).
  [MOI] 7 canaux de sequence, 8 scalaires de contexte (detail dans FEATURES).
  [MOI] normalisation robuste au niveau du PUITS (mediane/IQR), jamais par
        candidat: z-scorer chaque fenetre candidate separement detruirait
        justement la difference inter-candidats qu'on cherche a classer.
  [MOI] cible SOFT par defaut (voir `soft_target`), temperature APPRISE.
  [MOI] GroupNorm partout — intel (E) du corpus: le padding a longueur
        variable empoisonne les running stats de BatchNorm et FIGE
        silencieusement les predictions en mode eval. Plusieurs verdicts
        "l'aligneur n'apprend pas" du corpus etaient ce bug, pas le modele.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# torch est optionnel a l'IMPORT: tout le harnais numpy (features, metriques,
# bootstrap, banc de candidats) reste utilisable sans lui.
# ROGII_NO_TORCH=1 interdit meme l'IMPORT — pas seulement l'usage. Sous la
# contrainte CPU-ONLY de ce depot, "ne pas s'en servir" et "ne pas le charger"
# sont deux affirmations differentes, et seule la seconde est verifiable par
# `"torch" not in sys.modules`.
if os.environ.get("ROGII_NO_TORCH", "") not in ("", "0"):
    torch = nn = F = None
    HAVE_TORCH = False
else:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        HAVE_TORCH = True
    except Exception:  # pragma: no cover
        torch = nn = F = None
        HAVE_TORCH = False

# ---------------------------------------------------------------- constantes
SEQ_LEN = 512          # [MOI] longueur fixe de sequence
N_CHAN = 7             # canaux de sequence (voir build_features)
N_SCAL = 8             # scalaires de contexte par candidat
CTX_FT = (-3.0, 0.0, +3.0)   # contexte de profondeur du typewell (corpus: "+-3 ft")
GRID_FT = 60.0         # demi-largeur de la grille de scan, corpus S15 §5.3
CATASTROPHIC_FT = 12.0 # seuil "puits catastrophique" (README / corpus)
NEG_MASK = -1.0e4      # masque FINI des candidats de bourrage (jamais -inf:
                       # -inf * cible 0 => nan au backward, mesure au smoke)

CHAN_NAMES = ["gr_meas", "tw_at", "tw_minus3", "tw_plus3", "diff", "obs", "shape"]
SCAL_NAMES = ["cost_z", "cost_rel", "rank", "off_abs", "off_sign",
              "obs_frac", "toe_len", "n_cand"]


# ---------------------------------------------------------------- utilitaires
def block_resample(x: np.ndarray, L: int) -> np.ndarray:
    """Decimation par MOYENNE DE BLOC vers L points. [MOI]

    np.interp sur 4700 -> 512 aliase; la moyenne de bloc est le bon
    anti-aliasing. Si n < L on retombe sur le plus proche voisin.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.zeros(L, dtype=np.float64)
    if n == L:
        return x.copy()
    edges = np.linspace(0.0, n, L + 1)
    idx = np.clip(np.floor(edges).astype(np.int64), 0, n)
    cnt = idx[1:] - idx[:-1]
    cs = np.concatenate([[0.0], np.cumsum(x)])
    s = cs[idx[1:]] - cs[idx[:-1]]
    out = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
    empty = cnt <= 0
    if empty.any():  # cas n < L: plus proche voisin
        near = np.clip(idx[:-1][empty], 0, n - 1)
        out[empty] = x[near]
    return out


def robust_norm(x: np.ndarray, med: float, iqr: float) -> np.ndarray:
    return (x - med) / (iqr + 1e-6)


def robust_stats(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return float(med), float(max(q3 - q1, 1e-6))


# ---------------------------------------------------------------- banc de cand.
@dataclass
class CandidateBank:
    """Le banc de candidats. Cf. le CONTRAT NPZ en tete de module."""
    wells: np.ndarray
    fold: np.ndarray
    n_row: np.ndarray
    e_sum: np.ndarray
    e_sq: np.ndarray
    ptr: np.ndarray
    off: np.ndarray
    cost: np.ndarray
    gr: np.ndarray
    obs: np.ndarray
    shape: np.ndarray
    tw: np.ndarray
    obs_frac: np.ndarray
    toe_len: np.ndarray

    # -- proprietes
    @property
    def n_wells(self) -> int:
        return len(self.wells)

    @property
    def seq_len(self) -> int:
        return int(self.gr.shape[1])

    def n_cand(self, i: int) -> int:
        return int(self.ptr[i + 1] - self.ptr[i])

    def delta_true(self) -> np.ndarray:
        """D* = e_sum / n_row: le meilleur offset constant du puits (oracle O1).

        C'EST DE LA VERITE. Jamais sur un chemin d'inference: uniquement cible
        d'entrainement (folds train) et evaluation (hors-fold)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.e_sum / np.maximum(self.n_row, 1)

    # -- validation
    def check(self) -> None:
        W, L = self.n_wells, self.seq_len
        assert self.ptr.shape == (W + 1,), self.ptr.shape
        Kt = int(self.ptr[-1])
        for nm, arr, shp in [
            ("fold", self.fold, (W,)), ("n_row", self.n_row, (W,)),
            ("e_sum", self.e_sum, (W,)), ("e_sq", self.e_sq, (W,)),
            ("off", self.off, (Kt,)), ("cost", self.cost, (Kt,)),
            ("gr", self.gr, (W, L)), ("obs", self.obs, (W, L)),
            ("shape", self.shape, (W, L)), ("tw", self.tw, (Kt, L, 3)),
            ("obs_frac", self.obs_frac, (W,)), ("toe_len", self.toe_len, (W,)),
        ]:
            assert arr.shape == shp, (nm, arr.shape, shp)
        assert (np.diff(self.ptr) >= 1).all(), "un puits sans candidat"
        assert np.isfinite(self.off).all() and np.isfinite(self.cost).all()

    # -- io
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path, wells=self.wells, fold=self.fold, n_row=self.n_row,
            e_sum=self.e_sum, e_sq=self.e_sq, ptr=self.ptr, off=self.off,
            cost=self.cost, gr=self.gr, obs=self.obs, shape=self.shape,
            tw=self.tw, obs_frac=self.obs_frac, toe_len=self.toe_len)

    @staticmethod
    def load(path: str) -> "CandidateBank":
        d = np.load(path, allow_pickle=False)
        b = CandidateBank(**{k: d[k] for k in (
            "wells", "fold", "n_row", "e_sum", "e_sq", "ptr", "off", "cost",
            "gr", "obs", "shape", "tw", "obs_frac", "toe_len")})
        b.check()
        return b

    def subset(self, idx: np.ndarray) -> "CandidateBank":
        idx = np.asarray(idx, dtype=np.int64)
        parts = [np.arange(self.ptr[i], self.ptr[i + 1]) for i in idx]
        cidx = np.concatenate(parts) if parts else np.zeros(0, np.int64)
        ptr = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
        return CandidateBank(
            wells=self.wells[idx], fold=self.fold[idx], n_row=self.n_row[idx],
            e_sum=self.e_sum[idx], e_sq=self.e_sq[idx], ptr=ptr,
            off=self.off[cidx], cost=self.cost[cidx], gr=self.gr[idx],
            obs=self.obs[idx], shape=self.shape[idx], tw=self.tw[cidx],
            obs_frac=self.obs_frac[idx], toe_len=self.toe_len[idx])


# ---------------------------------------------------------------- features
def build_features(bank: CandidateBank, i: int, seq_len: int | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Construit (seq, scal) pour TOUS les candidats du puits i.

    seq  : (K, N_CHAN, L) float32
    scal : (K, N_SCAL)    float32

    Canaux [MOI], dans l'ordre de CHAN_NAMES:
      0 gr_meas   GR mesure le long du toe, normalise robuste AU PUITS
      1 tw_at     typewell lu a la profondeur du candidat (contexte 0 ft)
      2 tw_minus3 idem a -3 ft
      3 tw_plus3  idem a +3 ft
      4 diff      gr_meas - tw_at  (le residu d'appariement, explicite)
      5 obs       fraction de GR reellement observe dans le bloc
      6 shape     chemin de base, relatif a l'ancre, en unites de 30 ft

    Normalisation: mediane/IQR calculees UNE FOIS par puits (le GR mesure sur
    lui-meme; le typewell sur le canal 0 ft POOLE SUR TOUS LES CANDIDATS). On
    ne normalise JAMAIS par candidat: la moyenne de la fenetre typewell est une
    partie du signal de niveau, la retirer par candidat le detruirait.
    """
    a, b = int(bank.ptr[i]), int(bank.ptr[i + 1])
    K = b - a
    L0 = bank.seq_len
    L = seq_len or L0

    gr = bank.gr[i].astype(np.float64)
    obs = bank.obs[i].astype(np.float64)
    shp = bank.shape[i].astype(np.float64)
    tw = bank.tw[a:b].astype(np.float64)          # (K, L0, 3)

    if L != L0:
        gr = block_resample(gr, L)
        obs = block_resample(obs, L)
        shp = block_resample(shp, L)
        tw = np.stack([[block_resample(tw[k, :, c], L) for c in range(3)]
                       for k in range(K)])        # (K, 3, L)
        tw = np.transpose(tw, (0, 2, 1))          # (K, L, 3)

    g_med, g_iqr = robust_stats(gr)
    t_med, t_iqr = robust_stats(tw[:, :, 1])
    grn = robust_norm(gr, g_med, g_iqr)
    twn = robust_norm(tw, t_med, t_iqr)           # (K, L, 3)

    seq = np.empty((K, N_CHAN, L), dtype=np.float32)
    seq[:, 0] = grn[None, :]
    seq[:, 1] = twn[:, :, 1]
    seq[:, 2] = twn[:, :, 0]
    seq[:, 3] = twn[:, :, 2]
    seq[:, 4] = grn[None, :] - twn[:, :, 1]
    seq[:, 5] = obs[None, :]
    # ancre-relatif: la valeur au premier point du toe est retiree, comme dans
    # docs/23 ou `t` est le MD RELATIF A L'ANCRE.
    seq[:, 6] = ((shp - shp[0]) / 30.0)[None, :]
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

    cost = bank.cost[a:b].astype(np.float64)
    off = bank.off[a:b].astype(np.float64)
    c_mu, c_sd = float(cost.mean()), float(cost.std() + 1e-9)
    c_lo, c_hi = float(cost.min()), float(cost.max())
    rank = np.argsort(np.argsort(cost)).astype(np.float64)

    scal = np.empty((K, N_SCAL), dtype=np.float32)
    scal[:, 0] = (cost - c_mu) / c_sd
    scal[:, 1] = (cost - c_lo) / (c_hi - c_lo + 1e-9)
    scal[:, 2] = rank / max(K - 1, 1)
    scal[:, 3] = np.abs(off) / GRID_FT
    scal[:, 4] = off / GRID_FT
    scal[:, 5] = float(bank.obs_frac[i])
    scal[:, 6] = np.log1p(max(float(bank.toe_len[i]), 0.0)) / 9.0
    scal[:, 7] = K / 20.0
    scal = np.nan_to_num(scal, nan=0.0, posinf=0.0, neginf=0.0)
    return seq, scal


def soft_target(off: np.ndarray, delta_true: float, sigma: float = 3.0,
                hard: bool = False) -> np.ndarray:
    """Cible LISTWISE sur l'ensemble de candidats d'un puits.

    hard=True : one-hot sur le candidat le plus proche de la verite — la lettre
                du corpus ("cross-entropy against the candidate closest to the
                truth").
    hard=False (DEFAUT, [MOI]) : q_k proportionnel a exp(-|D_k - D*| / sigma).

    POURQUOI SOFT PAR DEFAUT. La lecture est une MOYENNE A POSTERIORI. Une
    cible one-hot pousse le softmax a saturer; a saturation la moyenne a
    posteriori DEGENERE en argmax et on perd exactement le gain de moyennage
    auquel le corpus attribue la victoire (et que notre propre dossier A6
    redit: sous RMSE, avec des hypotheses quasi ex aequo, le mode est la
    mauvaise statistique). sigma = 3 ft est calibre sur la bande +-5 ft dans
    laquelle le corpus place la verite par rapport au minimum local le plus
    proche (97 % des puits).
    """
    d = np.abs(np.asarray(off, dtype=np.float64) - float(delta_true))
    if hard:
        q = np.zeros_like(d)
        q[int(np.argmin(d))] = 1.0
        return q
    w = np.exp(-d / max(sigma, 1e-6))
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        q = np.zeros_like(d)
        q[int(np.argmin(d))] = 1.0
        return q
    return w / s


# ---------------------------------------------------------------- metriques
def sse_with_delta(n_row, e_sum, e_sq, delta) -> np.ndarray:
    """SSE du puits apres application d'un offset CONSTANT delta.

    pred' = pred_base + delta  =>  residu = e - delta, donc
        SSE(delta) = e_sq - 2*delta*e_sum + n*delta^2.
    Exact, aucun besoin de garder les lignes."""
    n = np.asarray(n_row, dtype=np.float64)
    return (np.asarray(e_sq, float) - 2.0 * np.asarray(delta, float)
            * np.asarray(e_sum, float) + n * np.asarray(delta, float) ** 2)


def pooled_rmse(n_row, e_sum, e_sq, delta) -> float:
    """RMSE POOLE par ligne — la metrique de la competition."""
    sse = np.maximum(sse_with_delta(n_row, e_sum, e_sq, delta), 0.0)
    return float(np.sqrt(sse.sum() / max(np.asarray(n_row, float).sum(), 1.0)))


def perwell_rmse(n_row, e_sum, e_sq, delta) -> np.ndarray:
    sse = np.maximum(sse_with_delta(n_row, e_sum, e_sq, delta), 0.0)
    return np.sqrt(sse / np.maximum(np.asarray(n_row, float), 1.0))


def oracle_delta(bank: CandidateBank) -> np.ndarray:
    """Pour chaque puits, l'offset du MEILLEUR candidat (selection oracle).

    Minimiser SSE(D_k) revient a minimiser |D_k - D*| : c'est le plafond
    atteignable par n'importe quel selecteur sur CET ensemble de candidats."""
    dt = bank.delta_true()
    out = np.zeros(bank.n_wells)
    for i in range(bank.n_wells):
        a, b = int(bank.ptr[i]), int(bank.ptr[i + 1])
        o = bank.off[a:b]
        out[i] = o[int(np.argmin(np.abs(o - dt[i])))]
    return out


def catastrophic_report(n_row, e_sum, e_sq, delta_base, delta_new,
                        thresh: float = CATASTROPHIC_FT) -> dict:
    """Puits catastrophiques (RMSE de base > thresh) et leur sort."""
    r0 = perwell_rmse(n_row, e_sum, e_sq, delta_base)
    r1 = perwell_rmse(n_row, e_sum, e_sq, delta_new)
    cat = r0 > thresh
    ncat = int(cat.sum())
    if ncat == 0:
        return {"n_catastrophic": 0, "frac_rescued": float("nan"),
                "frac_improved": float("nan"), "mean_gain_ft": float("nan"),
                "pooled_cat_before": float("nan"), "pooled_cat_after": float("nan")}
    resc = float((r1[cat] <= thresh).mean())
    impr = float((r1[cat] < r0[cat]).mean())
    n = np.asarray(n_row, float)
    return {
        "n_catastrophic": ncat,
        "frac_rescued": resc,
        "frac_improved": impr,
        "mean_gain_ft": float((r0[cat] - r1[cat]).mean()),
        "pooled_cat_before": float(np.sqrt((r0[cat] ** 2 * n[cat]).sum() / n[cat].sum())),
        "pooled_cat_after": float(np.sqrt((r1[cat] ** 2 * n[cat]).sum() / n[cat].sum())),
    }


def block_bootstrap_support(n_row, e_sum, e_sq, delta_a, delta_b,
                            n_boot: int = 1000, seed: int = 0) -> float:
    """Support bootstrap APPARIE PAR BLOCS DE PUITS pour "b bat a".

    Le corpus rapporte exactement cette statistique (support 1.000). On
    reechantillonne les PUITS (bloc = puits), pas les lignes: c'est la seule
    unite independante ici."""
    rng = np.random.default_rng(seed)
    n = np.asarray(n_row, float)
    sa = np.maximum(sse_with_delta(n_row, e_sum, e_sq, delta_a), 0.0)
    sb = np.maximum(sse_with_delta(n_row, e_sum, e_sq, delta_b), 0.0)
    W = len(n)
    win = 0
    for _ in range(n_boot):
        idx = rng.integers(0, W, W)
        d = n[idx].sum()
        if np.sqrt(sb[idx].sum() / d) < np.sqrt(sa[idx].sum() / d):
            win += 1
    return win / float(n_boot)


# ---------------------------------------------------------------- torch
if HAVE_TORCH:

    class LevelScorerCNN(nn.Module):
        """Encodeur CNN 1-D par candidat -> UN logit.

        NORMALISATION: GroupNorm exclusivement. Intel (E) du corpus: le padding
        a longueur variable empoisonne les running stats de BatchNorm, ce qui
        FIGE les predictions en mode eval pendant que le reseau apprend — c'est
        ce bug d'instrument, et non le modele, qui a produit plusieurs verdicts
        "l'aligneur n'entraine pas" dans le corpus. Aucune couche de ce module
        ne porte de statistique glissante.

        Le batch est (B, K, C, L) avec K variable par puits, aplati en
        (B*K, C, L) pour l'encodeur, puis re-plie pour le softmax LISTWISE. Les
        candidats de bourrage sont masques a -inf dans le softmax; ils ne
        touchent aucune statistique parce qu'il n'y en a aucune.
        """

        def __init__(self, n_chan: int = N_CHAN, n_scal: int = N_SCAL,
                     width: int = 48, depth: int = 4, ksize: int = 9,
                     groups: int = 8, head: int = 64, dropout: float = 0.10):
            super().__init__()
            layers = []
            cin = n_chan
            for d in range(depth):
                cout = width * (2 if d >= depth // 2 else 1)
                layers += [
                    nn.Conv1d(cin, cout, ksize, stride=2, padding=ksize // 2),
                    nn.GroupNorm(min(groups, cout), cout),
                    nn.GELU(),
                ]
                cin = cout
            self.enc = nn.Sequential(*layers)
            self.cout = cin
            self.head = nn.Sequential(
                nn.Linear(2 * cin + n_scal, head),
                nn.LayerNorm(head),          # LayerNorm: pas de running stat
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head, 1),
            )
            # temperature APPRISE [MOI]: elle fixe la nettete du posterior,
            # donc directement le compromis biais/variance de la moyenne a
            # posteriori. La laisser a 1 fige la statistique de lecture.
            self.log_tau = nn.Parameter(torch.zeros(1))

        def forward(self, seq, scal, cmask=None):
            """seq (B,K,C,L) · scal (B,K,S) -> logits BRUTS (B,K).

            ⚠ On ne masque PAS ici. Masquer avec -inf avant la division par la
            temperature APPRISE produit `0 * -inf = nan` au backward et NaN-ifie
            tout le reseau en un pas (mesure: tau=nan, loss=0.0000 des le smoke).
            Le masquage se fait apres l'echelle de temperature, avec une valeur
            FINIE — cf. `masked_logits`."""
            B, K, C, L = seq.shape
            h = self.enc(seq.reshape(B * K, C, L))
            h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
            h = torch.cat([h, scal.reshape(B * K, -1)], dim=-1)
            return self.head(h).reshape(B, K)

        def tau(self):
            return F.softplus(self.log_tau) + 1e-3

        def masked_logits(self, logits, cmask, scale: bool = True):
            z = logits / self.tau() if scale else logits
            return z.masked_fill(~cmask, NEG_MASK)

        def posterior(self, logits, cmask):
            return torch.softmax(self.masked_logits(logits, cmask), dim=-1) * cmask

        def readout(self, logits, off, cmask):
            """LECTURE = MOYENNE A POSTERIORI sur les offsets, PAS l'argmax.

            pred = sum_k softmax(s_k/tau) * (chemin + D_k); le chemin etant
            commun aux candidats, cela se reduit a chemin + sum_k p_k D_k."""
            p = self.posterior(logits, cmask)
            return (p * off).sum(dim=-1)

        def top1(self, logits, off, cmask):
            k = self.masked_logits(logits, cmask, scale=False).argmax(dim=-1)
            return off.gather(1, k[:, None])[:, 0]

    def listwise_loss(logits, target, cmask, tau=None):
        """Entropie croisee LISTWISE sur l'ensemble de candidats du puits.

        PAS de regression ponctuelle: le corpus est explicite, la valeur est
        dans le CLASSEMENT d'un ensemble discret."""
        z = logits / tau if tau is not None else logits
        z = z.masked_fill(~cmask, NEG_MASK)   # FINI: cf. forward()
        logp = torch.log_softmax(z, dim=-1)
        loss = -(target * logp).sum(dim=-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("perte listwise non finie — bug, pas du bruit")
        return loss

    def collate(items: list[dict], max_cand: int | None = None) -> dict:
        """Bourrage a Kmax + masque de candidats. Aucune stat glissante en jeu."""
        K = max(it["seq"].shape[0] for it in items)
        if max_cand:
            K = min(K, max_cand)
        B = len(items)
        C, L = items[0]["seq"].shape[1], items[0]["seq"].shape[2]
        S = items[0]["scal"].shape[1]
        seq = np.zeros((B, K, C, L), np.float32)
        scal = np.zeros((B, K, S), np.float32)
        off = np.zeros((B, K), np.float32)
        tgt = np.zeros((B, K), np.float32)
        msk = np.zeros((B, K), bool)
        for i, it in enumerate(items):
            k = min(it["seq"].shape[0], K)
            seq[i, :k] = it["seq"][:k]
            scal[i, :k] = it["scal"][:k]
            off[i, :k] = it["off"][:k]
            t = it["target"][:k]
            tgt[i, :k] = t / max(t.sum(), 1e-9)
            msk[i, :k] = True
        return {
            "seq": torch.from_numpy(seq), "scal": torch.from_numpy(scal),
            "off": torch.from_numpy(off), "target": torch.from_numpy(tgt),
            "cmask": torch.from_numpy(msk),
            "widx": np.array([it["widx"] for it in items], np.int64),
        }


class LevelScorerDataset:
    """Un item = UN PUITS et TOUT son ensemble de candidats (unite listwise).

    `train=True` exige `delta_true` fini (folds d'entrainement). En inference
    la cible n'est jamais touchee.
    """

    def __init__(self, bank: CandidateBank, widx: np.ndarray, seq_len: int = SEQ_LEN,
                 sigma: float = 3.0, hard: bool = False, max_cand: int | None = None,
                 train: bool = True, augment: bool = False, seed: int = 0):
        self.bank = bank
        self.widx = np.asarray(widx, np.int64)
        self.seq_len = seq_len
        self.sigma = sigma
        self.hard = hard
        self.max_cand = max_cand
        self.train = train
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self._dt = bank.delta_true()
        if train:
            keep = np.isfinite(self._dt[self.widx])
            self.widx = self.widx[keep]

    def __len__(self):
        return len(self.widx)

    def __getitem__(self, j: int) -> dict:
        i = int(self.widx[j])
        seq, scal = build_features(self.bank, i, self.seq_len)
        a, b = int(self.bank.ptr[i]), int(self.bank.ptr[i + 1])
        off = self.bank.off[a:b].astype(np.float64)
        if self.augment:
            # [MOI] augmentation legere: bruit de gain sur les canaux GR et
            # inversion du sens du toe. Aucune ne bouge l'offset.
            seq = seq * np.float32(1.0 + 0.05 * self.rng.standard_normal())
            if self.rng.random() < 0.5:
                seq = seq[:, :, ::-1].copy()
        if self.max_cand and len(off) > self.max_cand:
            # on garde les candidats de plus BAS cout (le contrat du corpus)
            sel = np.argsort(self.bank.cost[a:b])[: self.max_cand]
            sel = np.sort(sel)
            seq, scal, off = seq[sel], scal[sel], off[sel]
        dt = self._dt[i]
        tgt = (soft_target(off, dt, self.sigma, self.hard) if np.isfinite(dt)
               else np.full(len(off), 1.0 / len(off)))
        return {"seq": seq, "scal": scal, "off": off.astype(np.float32),
                "target": tgt.astype(np.float32), "widx": i}


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ------------------------------------------------- lecture sans torch (controle)
def local_posterior_mean(off: np.ndarray, logit: np.ndarray,
                         radius: float = 8.0) -> float:
    """Moyenne a posteriori RESTREINTE au voisinage du mode dominant. [MOI]

    MESURE (smoke, synthetique a selection imparfaite): la moyenne a posteriori
    PLEINE perd contre le top-1 des que les modes alias sont ELOIGNES — moyenner
    entre un mode juste a 0 ft et un mode faux a 30 ft atterrit a 15 ft, pire que
    de trancher. Le gain de couverture du corpus vient des quasi-ex-aequo
    PROCHES, pas des alias lointains. Cette lecture garde le moyennage la ou il
    paie (dans le mode) et la commet la ou il coute (entre modes).

    DIAGNOSTIC: la porte pre-enregistree reste sur la moyenne a posteriori
    PLEINE, telle que le corpus la specifie."""
    off = np.asarray(off, float)
    z = np.asarray(logit, float)
    k = int(np.argmax(z))
    sel = np.abs(off - off[k]) <= radius
    o, zz = off[sel], z[sel]
    zz = zz - zz.max()
    p = np.exp(zz)
    return float((p * o).sum() / max(p.sum(), 1e-12))


def numpy_readout(off: np.ndarray, logit: np.ndarray, tau: float = 1.0
                  ) -> tuple[float, float]:
    """(moyenne a posteriori, top-1) — reference numpy de la lecture torch.

    Sert de controle bit-a-bit du calcul de lecture et permet de faire tourner
    tout le harnais de metriques SANS torch."""
    z = np.asarray(logit, float) / max(tau, 1e-6)
    z = z - z.max()
    p = np.exp(z)
    p = p / max(p.sum(), 1e-12)
    return float((p * off).sum()), float(off[int(np.argmax(logit))])
