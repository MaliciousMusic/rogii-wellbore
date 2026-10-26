"""rogii.seqmodel — U-Net 1D sur la sequence complete d'un puits.

Ce que le reseau predit: g = dS/dMD, le PENDAGE geologique, et rien d'autre.
La reconstruction
        delta(i) = -(Z_i - Z_ps) + somme_{ps<j<=i} g_j * DS
est faite HORS du reseau, ce qui garantit:
  * continuite exacte au point PS (aucune prediction ne peut sauter),
  * la physique  dTVT = -dZ + dS  imposee par construction,
  * une sortie lisse par nature (integrale d'une quantite bornee), ce qui
    correspond au fait mesure que la verite est une polyligne a ~7 segments.

fp32 uniquement: le forum rapporte que bfloat16 casse silencieusement ces
modeles ici (echec meme sur une tache identite, sans NaN ni crash).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ngroups(c, want=8):
    """plus grand diviseur de c inferieur ou egal a `want`.

    GroupNorm exige num_channels % num_groups == 0. Prendre min(want, c) casse
    des que la largeur de base n'est pas un multiple de 8 (ex. base=20).
    """
    for g in range(min(want, c), 0, -1):
        if c % g == 0:
            return g
    return 1


class MaskedGroupNorm(nn.Module):
    """GroupNorm dont les statistiques (mean/var) sont calculees UNIQUEMENT sur
    les positions VALIDES (mask=1); le zero-padding en est exclu. Hors positions
    valides, la sortie est remise a 0.

    C'est le coeur du correctif B1 (docs/12): les sequences sont paddees a LMAX
    avec ~46,6 % de zeros VARIABLE par puits (L 653-2951/3072). Une GroupNorm
    ordinaire calcule mean/var sur TOUT LMAX, zeros de padding inclus, donc le
    facteur de re-echelle depend de la longueur du puits — une nuisance injectee
    a chaque couche. Ici la normalisation ne voit que le vrai signal.

    Semantique identique a nn.GroupNorm (affine PAR CANAL, meme parametres
    weight/bias) quand mask==1 partout, aux erreurs d'arrondi pres.

    mask: (B, 1, L) ou (B, L), 1 = position reelle, 0 = padding.
    """

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.affine = bool(affine)
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x, mask):
        B, C, L = x.shape
        G = self.num_groups
        cg = C // G
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)                       # (B, 1, L)
        m = mask.reshape(B, 1, 1, L).to(x.dtype)           # broadcast (B,1,1,L)
        xr = x.reshape(B, G, cg, L)
        # nombre d'elements valides par groupe = n_valid * cg (mask commun aux
        # canaux d'un groupe). clamp evite la division par zero (puits vide).
        denom = (m.sum(dim=(2, 3), keepdim=True) * cg).clamp_min(1.0)  # (B,1,1,1)
        mean = (xr * m).sum(dim=(2, 3), keepdim=True) / denom          # (B,G,1,1)
        var = (((xr - mean) ** 2) * m).sum(dim=(2, 3), keepdim=True) / denom
        xn = (xr - mean) / torch.sqrt(var + self.eps)
        xn = xn.reshape(B, C, L)
        if self.affine:
            xn = xn * self.weight.reshape(1, C, 1) + self.bias.reshape(1, C, 1)
        return xn * mask.reshape(B, 1, L).to(x.dtype)      # zero hors valide


def _downsample_mask(m, target_len):
    """Sous-echantillonne le masque de validite (B,1,L) a `target_len` de facon
    coherente avec le pooling stride-2 du U-Net (Conv1d k=4 s=2 p=1): une cellule
    aval est VALIDE des qu'une position amont valide tombe dans son support.
    max_pool aux memes k/s/p reproduit exactement la geometrie du pooling conv."""
    m = F.max_pool1d(m, kernel_size=4, stride=2, padding=1)
    if m.shape[-1] == target_len:
        return m
    if m.shape[-1] > target_len:
        return m[..., :target_len]
    return F.pad(m, (0, target_len - m.shape[-1]))


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=5, dil=1, groups=8, masknorm=False):
        super().__init__()
        pad = dil * (k - 1) // 2
        self.masknorm = bool(masknorm)
        ng = _ngroups(cout, groups)
        Norm = MaskedGroupNorm if masknorm else nn.GroupNorm
        # Ordre de creation des modules PRESERVE (c1, n1, c2, n2, skip) pour que
        # le tirage RNG d'init soit identique flag ON/OFF: seules les Conv1d
        # consomment le RNG, GroupNorm et MaskedGroupNorm initialisent a 1/0.
        self.c1 = nn.Conv1d(cin, cout, k, padding=pad, dilation=dil)
        self.n1 = Norm(ng, cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=pad, dilation=dil)
        self.n2 = Norm(ng, cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, mask=None):
        if not (self.masknorm and mask is not None):
            # --- chemin historique EXACT (bit-a-bit) ---
            h = F.gelu(self.n1(self.c1(x)))
            h = self.n2(self.c2(h))
            return F.gelu(h + self.skip(x))
        # --- chemin MASQUE: stats hors-padding + re-zero des positions invalides
        # (les zeros de padding saignent dans les positions valides via le noyau
        # aux frontieres; re-zeroer en fin de bloc empeche l'accumulation).
        h = F.gelu(self.n1(self.c1(x), mask))
        h = self.n2(self.c2(h), mask)
        out = F.gelu(h + self.skip(x))
        return out * mask


class UNet1D(nn.Module):
    """out_smooth: moyenne glissante appliquee A LA SORTIE.

    Justification mesuree (eda08): dans la zone d'evaluation la verite se
    decrit avec ~7 segments lineaires a 2 ft pres sur 4 840 ft. Elle n'a donc
    aucun contenu haute frequence. Lisser la sortie ne detruit rien de la
    cible et supprime du bruit de prediction pur — lisser la VERITE a 100 ft
    ne coute que 0.139 ft de RMSE.
    """

    def __init__(self, cin=18, base=48, depth=6, out_scale=0.06, out_smooth=25,
                 masknorm=False):
        super().__init__()
        self.out_smooth = out_smooth
        self.masknorm = bool(masknorm)
        chs = [base * (2 ** min(i, 3)) for i in range(depth)]
        self.stem = ConvBlock(cin, chs[0], masknorm=masknorm)
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        for i in range(depth - 1):
            self.pool.append(nn.Conv1d(chs[i], chs[i], 4, stride=2, padding=1))
            self.down.append(ConvBlock(chs[i], chs[i + 1], masknorm=masknorm))
        self.mid = ConvBlock(chs[-1], chs[-1], dil=2, masknorm=masknorm)
        self.up = nn.ModuleList()
        self.upc = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up.append(nn.ConvTranspose1d(chs[i], chs[i - 1], 4, stride=2, padding=1))
            self.upc.append(ConvBlock(chs[i - 1] * 2, chs[i - 1], masknorm=masknorm))
        self.head = nn.Conv1d(chs[0], 1, 1)
        self.out_scale = out_scale
        # tete a zero: la sortie initiale vaut exactement la baseline plate
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, return_feat=False, mask=None):
        """return_feat=False (defaut) : contrat historique EXACT (renvoie o).
        return_feat=True : renvoie AUSSI la carte de features pre-tete `feat`
        (chs[0] canaux, pleine resolution) pour brancher des tetes auxiliaires
        sur le MEME backbone sans toucher a la sortie principale.

        mask (B,1,L) ou (B,L) : masque de validite, 1 sur les positions reelles.
        Ignore sauf si masknorm=True (correctif B1). Le sous-echantillonnage du
        masque suit exactement la geometrie du pooling stride-2."""
        if not (self.masknorm and mask is not None):
            # --- chemin historique EXACT (bit-a-bit) ---
            h = self.stem(x)
            skips = [h]
            for pool, blk in zip(self.pool, self.down):
                h = blk(pool(h))
                skips.append(h)
            h = self.mid(h)
            for i, (up, blk) in enumerate(zip(self.up, self.upc)):
                h = up(h)
                s = skips[-(i + 2)]
                if h.shape[-1] != s.shape[-1]:
                    h = F.pad(h, (0, s.shape[-1] - h.shape[-1]))
                h = blk(torch.cat([h, s], dim=1))
            # tete LINEAIRE, pas tanh. Mesure (diag_seq2): avec tanh*120 la
            # pre-activation sature sur TOUTE la sequence, la sortie devient
            # constante par puits, donc (u - u_ps) = 0 exactement et le gradient
            # meurt. Le modele restait gele sur la baseline plate a la 4e decimale.
            # Init a zero => sortie initiale nulle => baseline plate exacte, sans
            # aucun plafond saturable.
            feat = h
            o = self.head(h) * self.out_scale
            if self.out_smooth and self.out_smooth > 1:
                k = int(self.out_smooth) | 1
                o = F.avg_pool1d(F.pad(o, (k // 2, k // 2), mode="replicate"), k, stride=1)
            if return_feat:
                return o.squeeze(1), feat
            return o.squeeze(1)
        # --- chemin MASQUE (correctif B1) ---
        if mask.dim() == 2:
            mask = mask[:, None]                          # (B, 1, L)
        mask = mask.to(x.dtype)
        h = self.stem(x, mask)
        skips = [h]
        masks = [mask]
        m = mask
        for pool, blk in zip(self.pool, self.down):
            h = pool(h)
            m = _downsample_mask(m, h.shape[-1])
            h = h * m                                     # zero-pad avant le bloc
            h = blk(h, m)
            skips.append(h)
            masks.append(m)
        h = self.mid(h, m)
        for i, (up, blk) in enumerate(zip(self.up, self.upc)):
            h = up(h)
            s = skips[-(i + 2)]
            ms = masks[-(i + 2)]
            if h.shape[-1] != s.shape[-1]:
                h = F.pad(h, (0, s.shape[-1] - h.shape[-1]))
            h = h * ms
            h = blk(torch.cat([h, s], dim=1), ms)
        feat = h
        o = self.head(h) * self.out_scale
        if self.out_smooth and self.out_smooth > 1:
            k = int(self.out_smooth) | 1
            o = F.avg_pool1d(F.pad(o, (k // 2, k // 2), mode="replicate"), k, stride=1)
        if return_feat:
            return o.squeeze(1), feat
        return o.squeeze(1)


class UNet1DMix(nn.Module):
    """Variante MIXTURE a K tetes du UNet1D — backbone strictement identique.

    head : Conv1d(chs[0], K, 1) — K hypotheses de delta en parallele.
    gate : pooling global MOYEN sur le bottleneck (sortie de self.mid)
           -> Linear -> logits K ; softmax = poids par puits pi_k.

    Init : poids des K tetes a zero (chaque tete part de la baseline plate,
    meme philosophie que UNet1D) ; biais a -15/0/+15 ft en unites de sortie
    AVANT out_scale, donc bias_k = (-15, 0, +15)/out_scale.
    NB mesure a l'implantation : avec reconstruct_direct (delta = u - u[i0])
    un biais CONSTANT s'annule exactement dans delta — cette init n'oriente
    donc PAS les sorties initiales (toutes valent la baseline plate). La
    brisure de symetrie effective vient du gate (init aleatoire PyTorch par
    defaut) : pi_k differe par puits, donc les poids de la soft-min et les
    gradients recus par chaque tete different des le premier pas.

    forward -> (o, logits) : o (B, K, L) deja mis a l'echelle et lisse,
    logits (B, K) AVANT softmax.
    """

    def __init__(self, cin=18, base=48, depth=6, out_scale=0.06, out_smooth=25,
                 K=3, bias_span_ft=15.0, masknorm=False):
        super().__init__()
        self.K = K
        self.out_smooth = out_smooth
        self.masknorm = bool(masknorm)
        chs = [base * (2 ** min(i, 3)) for i in range(depth)]
        self.stem = ConvBlock(cin, chs[0], masknorm=masknorm)
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        for i in range(depth - 1):
            self.pool.append(nn.Conv1d(chs[i], chs[i], 4, stride=2, padding=1))
            self.down.append(ConvBlock(chs[i], chs[i + 1], masknorm=masknorm))
        self.mid = ConvBlock(chs[-1], chs[-1], dil=2, masknorm=masknorm)
        self.up = nn.ModuleList()
        self.upc = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up.append(nn.ConvTranspose1d(chs[i], chs[i - 1], 4, stride=2, padding=1))
            self.upc.append(ConvBlock(chs[i - 1] * 2, chs[i - 1], masknorm=masknorm))
        self.head = nn.Conv1d(chs[0], K, 1)
        self.out_scale = out_scale
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.linspace(-bias_span_ft, bias_span_ft, K)
                                 / out_scale)
        # gate garde l'init aleatoire par defaut: c'est le brise-symetrie
        self.gate = nn.Linear(chs[-1], K)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, mask=None):
        """mask (B,1,L)/(B,L) : ignore sauf si masknorm=True (correctif B1)."""
        if not (self.masknorm and mask is not None):
            # --- chemin historique EXACT (bit-a-bit) ---
            h = self.stem(x)
            skips = [h]
            for pool, blk in zip(self.pool, self.down):
                h = blk(pool(h))
                skips.append(h)
            h = self.mid(h)
            logits = self.gate(h.mean(dim=2))              # (B, K)
            for i, (up, blk) in enumerate(zip(self.up, self.upc)):
                h = up(h)
                s = skips[-(i + 2)]
                if h.shape[-1] != s.shape[-1]:
                    h = F.pad(h, (0, s.shape[-1] - h.shape[-1]))
                h = blk(torch.cat([h, s], dim=1))
            o = self.head(h) * self.out_scale              # (B, K, L)
            if self.out_smooth and self.out_smooth > 1:
                k = int(self.out_smooth) | 1
                o = F.avg_pool1d(F.pad(o, (k // 2, k // 2), mode="replicate"), k, stride=1)
            return o, logits
        # --- chemin MASQUE (correctif B1) ---
        if mask.dim() == 2:
            mask = mask[:, None]
        mask = mask.to(x.dtype)
        h = self.stem(x, mask)
        skips = [h]
        masks = [mask]
        m = mask
        for pool, blk in zip(self.pool, self.down):
            h = pool(h)
            m = _downsample_mask(m, h.shape[-1])
            h = h * m
            h = blk(h, m)
            skips.append(h)
            masks.append(m)
        h = self.mid(h, m)
        # gate: moyenne sur les positions VALIDES seulement (pas h.mean sur tout
        # LMAX, zeros de padding inclus). m est le masque a la resolution du
        # bottleneck.
        denom = m.sum(dim=2).clamp_min(1.0)                # (B, 1)
        logits = self.gate((h * m).sum(dim=2) / denom)     # (B, K)
        for i, (up, blk) in enumerate(zip(self.up, self.upc)):
            h = up(h)
            s = skips[-(i + 2)]
            ms = masks[-(i + 2)]
            if h.shape[-1] != s.shape[-1]:
                h = F.pad(h, (0, s.shape[-1] - h.shape[-1]))
            h = h * ms
            h = blk(torch.cat([h, s], dim=1), ms)
        o = self.head(h) * self.out_scale                  # (B, K, L)
        if self.out_smooth and self.out_smooth > 1:
            k = int(self.out_smooth) | 1
            o = F.avg_pool1d(F.pad(o, (k // 2, k // 2), mode="replicate"), k, stride=1)
        return o, logits


class CorrEncoder(nn.Module):
    """Encodeur siamois 1D (poids partages) pour le volume de correlation.

    Entree : 4 canaux GR normalises talon — [gr, gr_s13, gr_s51, obs] — soit
    les canaux 7-10 du lateral, soit le profil typewell etire (memes canaux,
    construits par WellSeq.build_tw_pyramid + tensors).

    Trunk : 3 convolutions denses k=5 puis UNE convolution dilatee k=7 dil=4.
    Champ receptif : 9 mailles (36 ft) au tap FIN, 37 mailles (148 ft) au tap
    GROSSIER — dans la bande 130-150 ft demandee, plafonne sous 170 ft.
    (k=7 sur la couche dilatee: avec k=5 le RF ne serait que 116 ft.)

    Sortie : hypercolonne a 2 groupes de dim dg — concat(proj(fine),
    proj(coarse)) + un tap DIRECT 1x1 depuis l'entree.

    PAS de normalisation interne : GroupNorm/LayerNorm sur sequences paddees
    ferait dependre le descripteur d'un puits de la composition du batch
    (les stats par echantillon incluraient le padding, variable). Les entrees
    sont deja normalisees-talon et clippees a +-5.

    WARM START : projections a zero, tap direct = copie de gr_s13 (canal 1)
    dans la coordonnee 0 de CHAQUE groupe. A l'init le descripteur est donc
    exactement (gr_s13, 0, ..) et le volume (voir UNet1DCorr) reproduit
    l'emission gaussienne faite main.
    """

    def __init__(self, cin=4, width=24, dg=8):
        super().__init__()
        self.dg = dg
        self.c1 = nn.Conv1d(cin, width, 5, padding=2)
        self.c2 = nn.Conv1d(width, width, 5, padding=2)
        self.c3 = nn.Conv1d(width, width, 5, padding=2)
        self.cd = nn.Conv1d(width, width, 7, padding=12, dilation=4)
        self.pf = nn.Conv1d(width, dg, 1)      # projection du tap FIN
        self.pc = nn.Conv1d(width, dg, 1)      # projection du tap GROSSIER
        self.direct = nn.Conv1d(cin, 2 * dg, 1, bias=False)
        nn.init.zeros_(self.pf.weight); nn.init.zeros_(self.pf.bias)
        nn.init.zeros_(self.pc.weight); nn.init.zeros_(self.pc.bias)
        with torch.no_grad():
            self.direct.weight.zero_()
            self.direct.weight[0, 1, 0] = 1.0        # groupe fin    <- gr_s13
            self.direct.weight[dg, 1, 0] = 1.0       # groupe gros   <- gr_s13

    def forward(self, x):
        h = F.gelu(self.c1(x))
        hf = F.gelu(self.c2(h))                      # tap fin (RF 36 ft)
        h = F.gelu(self.c3(hf))
        h = F.gelu(self.cd(h))                       # tap grossier (RF 148 ft)
        return torch.cat([self.pf(hf), self.pc(h)], dim=1) + self.direct(x)


class UNet1DCorr(nn.Module):
    """U-Net + VOLUME DE CORRELATION APPRIS (tete --head corr).

    Remplace les canaux 18-38 (emission gaussienne faite main de seqdata) par
        C(m, delta, s) = corr( emb_lat(m), emb_tw_s(TVT_flat(m) + delta) )
    ou emb_tw_s est le descripteur du typewell ETIRE d'un facteur sigma_s
    (S=5 log-espaces, centres par puits sur le prior dz — cf. seqdata).

    La correlation est le produit scalaire corrige des normes
        c = -0.5 * sw * (|e_l|^2 + |e_t|^2 - 2 <e_l, e_t>)
    puis vol = exp(c) dans (0, 1]. C'est la forme <.,.> de la spec, avec les
    normes en biais appris — choisie parce qu'elle warm-starte EXACTEMENT:
    a l'init e = (gr_s13, 0, ..) et sw = (sd/sd_e)^2, donc
        vol = exp(-((gr_s1 - tw)/sd_e)^2 / 2) = l'emission actuelle.
    Reduction sur s par MAX (la spec autorisait max ou softmax appris).
    Gating identique a la version faite main: couverture GR (canal 10) et
    fenetre typewell -> 0.

    Injection: 21 offsets x 2 groupes d'hypercolonne = 42 canaux, concat aux
    canaux 0-17 -> trunk UNet1D(cin=18+42) inchange (tete a zero comprise:
    la sortie initiale reste la baseline plate exacte).

    forward(x, kit) -> (B, L), meme contrat de sortie que UNet1D.
    kit (voir make_batch de train_seq): pyr (B,S,4,T) fp32, h (B,S), t0 (B,S),
    lo/hi (B,), base (B,L), sw (B,).
    """

    def __init__(self, cin=18, base=48, depth=6, out_scale=0.06, out_smooth=25,
                 enc_width=24, enc_dg=8, n_off=21, off_step=8.0, aux=False,
                 masknorm=False):
        super().__init__()
        self.n_off = n_off
        self.n_groups = 2
        self.aux = aux
        self.out_scale = out_scale
        self.masknorm = bool(masknorm)
        self.enc = CorrEncoder(4, enc_width, enc_dg)
        self.register_buffer(
            "offsets", (torch.arange(n_off, dtype=torch.float32)
                        - n_off // 2) * off_step)
        self.unet = UNet1D(cin=cin + n_off * self.n_groups, base=base,
                           depth=depth, out_scale=out_scale,
                           out_smooth=out_smooth, masknorm=masknorm)
        # --- TETES AUXILIAIRES (playbook JP #3/#4), flag --aux -------------
        # Deux tetes 1x1 LEGERES sur le meme backbone (features pre-tete du
        # U-Net, chs[0] = base). Elles ne servent qu'a REGULARISER: l'inference
        # ignore return_aux et garde la sortie principale, bit-a-bit identique.
        #   head_dip   : pendage instantane dTVT/dMD (l'inconnu exact)
        #   head_delta : delta physique, contraint a coincider avec l'integrale
        #                du pendage (coherence differentiable).
        # ⚠ Creees APRES self.unet et SEULEMENT si aux: le tirage RNG des
        # organes partages (enc + unet) est donc identique avec ou sans aux,
        # d'ou la parite bit-a-bit du chemin --head corr sans --aux. Zero-init:
        # a l'init pendage=0 et delta_phys=0 => termes aux = mse(0, cible) et 0.
        if self.aux:
            self.head_dip = nn.Conv1d(base, 1, 1)
            self.head_delta = nn.Conv1d(base, 1, 1)
            for hh in (self.head_dip, self.head_delta):
                nn.init.zeros_(hh.weight)
                nn.init.zeros_(hh.bias)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def corr_volume(self, x, kit):
        B, _, L = x.shape
        dg, O = self.enc.dg, self.n_off
        lat = self.enc(x[:, 7:11])                        # (B, 2*dg, L)
        pyr = kit["pyr"]                                  # (B, S, 4, T)
        S, T = pyr.shape[1], pyr.shape[3]
        e_tw = self.enc(pyr.reshape(B * S, 4, T))         # (B*S, 2*dg, T)
        # requetes tau(b, o, l) = base + offset, en index de grille etiree
        tau = kit["base"][:, None, :] + self.offsets[None, :, None]  # (B,O,L)
        idx = (tau[:, None] - kit["t0"][:, :, None, None]) \
            / kit["h"][:, :, None, None]                  # (B, S, O, L)
        xg = idx.reshape(B * S, O, L) / max(T - 1, 1) * 2.0 - 1.0
        grid = torch.stack([xg, torch.zeros_like(xg)], dim=-1)
        samp = F.grid_sample(e_tw.unsqueeze(2), grid, mode="bilinear",
                             padding_mode="border", align_corners=True)
        e_l = lat.reshape(B, self.n_groups, dg, L)
        e_t = samp.reshape(B, S, self.n_groups, dg, O, L)
        dot = torch.einsum("bgdl,bsgdol->bsgol", e_l, e_t)
        d2 = (e_l ** 2).sum(2)[:, None, :, None, :] \
            + (e_t ** 2).sum(3) - 2.0 * dot               # (B, S, G, O, L)
        c = (-0.5 * kit["sw"][:, None, None, None, None] * d2).clamp(-30.0, 0.0)
        vol = torch.exp(c).max(dim=1).values              # max sur s (B,G,O,L)
        inr = ((tau >= kit["lo"][:, None, None])
               & (tau <= kit["hi"][:, None, None])).float()      # (B, O, L)
        vol = vol * inr[:, None] * x[:, 10].clamp(0.0, 1.0)[:, None, None, :]
        return vol.reshape(B, self.n_groups * O, L)

    def forward(self, x, kit, return_aux=False):
        """return_aux=False (defaut) : contrat historique EXACT -> (B, L).
        return_aux=True ET self.aux : renvoie (o, dip, delta_phys) ou o est la
        sortie principale INCHANGEE, dip = dTVT/dMD predit, delta_phys = delta
        predit directement par la tete physique. Utilise a l'ENTRAINEMENT
        seulement (regularisation); l'inference n'appelle jamais return_aux."""
        vol = self.corr_volume(x, kit)
        xin = torch.cat([x, vol], dim=1)
        # correctif B1: le masque de validite (1 sur [0,L), 0 sinon) est fourni
        # via kit["v"] (make_batch). None si masknorm OFF -> chemin bit-identique.
        mask = kit.get("v") if self.masknorm else None
        if not (return_aux and self.aux):
            return self.unet(xin, mask=mask)
        o, feat = self.unet(xin, return_feat=True, mask=mask)
        dip = (self.head_dip(feat) * self.out_scale).squeeze(1)
        delta_phys = (self.head_delta(feat) * self.out_scale).squeeze(1)
        return o, dip, delta_phys


class UNet1DOffset(UNet1DCorr):
    """Tete --head offset : sortie DISTRIBUTIONNELLE de l'OFFSET par point.

    Meme backbone U-Net 1D et MEME volume de correlation appris que UNet1DCorr
    (herite CorrEncoder `enc`, buffer `offsets`, methode `corr_volume`). La
    SEULE difference est la tete : au lieu d'un scalaire delta par point, on
    predit K_BINS logits par position ; softmax = distribution p(delta | i) sur
    des bins d'offset regulierement espaces ; la prediction ponctuelle est la
    MOYENNE A POSTERIORI (soft-argmin / soft-mean, GC-Net)

        delta(i) = sum_k bin_k * p(delta = bin_k | i)

    qui est l'estimateur RMSE-optimal de la distribution. Motivation (memoire
    21/07, croise Wiener) : predire le pendage dS/dMD puis l'INTEGRER accumule
    le bruit en niveau ; predire l'offset ABSOLU par point l'evite, et la sortie
    distributionnelle donne GRATUITEMENT une incertitude par point (entropie).

    QUE PORTENT LES BINS — mesure, pas supposition (scratchpad measure_geo.py,
    40 puits, zone d'eval) :
      * -(Z - Z_ps) a une amplitude MEDIANE 157 ft (max 252), et le residu de
        surface S - S_ps deborde +-80 ft sur 48.8 % des points : des bins +-80
        ne peuvent PAS representer S - S_ps, et delta = -(Z-Z_ps) (init offset
        nul de cette lecture-la) coute RMSE ~88 ft.
      * delta = TVT_i - TVT_ps reste |max| median 19.5 ft, 0.00 % au-dela de
        +-80 : c'est la BONNE grandeur pour des bins +-80, coherent avec le
        volume de correlation de seqdata qui balaie +-80 ft autour de la meme
        baseline TVT-plate (delta = 0).
    Les bins portent donc le DELTA relatif a la baseline TVT-plate (delta = 0).
    delta_final(i) = soft_mean_offset(i). Init a logits nuls => softmax
    UNIFORME symetrique => soft-mean = 0 => sortie = baseline "tenir la derniere
    TVT" (RMSE 15.91), exactement comme les tetes single/mix/corr (et NON la
    baseline S-constant a ~107 ft).

    Continuite PS : le forward renvoie le champ delta BRUT (B, L) — contrat de
    sortie identique a UNet1DCorr ; l'ancrage exact a 0 au PS est fait par
    reconstruct_direct (soustraction de la valeur au PS) dans eval_fold et dans
    la boucle d'entrainement, comme pour toutes les autres tetes.

    forward(x, kit)                 -> (B, L) delta        (contrat historique)
    forward(x, kit, return_dist=True) -> (delta, logits, p, entropy) :
        logits (B, K, L) AVANT softmax (pour la CE) ; p (B, K, L) ; entropy
        (B, L) en nats (uniforme = log K = log 81 = 4.394). Utilise a
        l'ENTRAINEMENT (perte CE) et pour l'incertitude OOF.
    """

    def __init__(self, cin=18, base=48, depth=6, out_scale=0.06, out_smooth=0,
                 enc_width=24, enc_dg=8, n_off=21, off_step=8.0,
                 k_bins=81, bin_step=2.0, masknorm=False):
        super().__init__(cin=cin, base=base, depth=depth, out_scale=out_scale,
                         out_smooth=out_smooth, enc_width=enc_width,
                         enc_dg=enc_dg, n_off=n_off, off_step=off_step, aux=False,
                         masknorm=masknorm)
        self.k_bins = int(k_bins)
        self.bin_step = float(bin_step)
        # out_smooth de la tete offset : defaut 0 (audit 09 L1/L2 — le reseau est
        # SOUS-lisse, ne pas re-lisser la sortie). L'out_smooth du U-Net interne
        # ne touche que sa sortie scalaire (inutilisee ici : on lit `feat`).
        self.off_smooth = int(out_smooth)
        self.register_buffer(
            "bins", (torch.arange(self.k_bins, dtype=torch.float32)
                     - (self.k_bins // 2)) * self.bin_step)   # -80 .. +80 ft
        self.head_bins = nn.Conv1d(base, self.k_bins, 1)
        # init logits nuls => softmax uniforme => soft-mean = 0 => baseline plate
        nn.init.zeros_(self.head_bins.weight)
        nn.init.zeros_(self.head_bins.bias)

    def forward(self, x, kit, return_dist=False):
        vol = self.corr_volume(x, kit)
        xin = torch.cat([x, vol], dim=1)
        mask = kit.get("v") if self.masknorm else None       # correctif B1
        _o, feat = self.unet(xin, return_feat=True, mask=mask)  # on n'utilise QUE feat
        logits = self.head_bins(feat)                        # (B, K, L)
        p = torch.softmax(logits, dim=1)                     # (B, K, L)
        offset = torch.einsum("bkl,k->bl", p, self.bins)     # soft-mean (B, L)
        if self.off_smooth and self.off_smooth > 1:
            k = int(self.off_smooth) | 1
            offset = F.avg_pool1d(
                F.pad(offset[:, None], (k // 2, k // 2), mode="replicate"),
                k, stride=1).squeeze(1)
        if return_dist:
            ent = -(p * torch.log(p.clamp_min(1e-9))).sum(dim=1)   # (B, L) nats
            return offset, logits, p, ent
        return offset


def reconstruct(g, i0, ds, valid):
    """delta(i) = somme_{ps<j<=i} g_j * ds,  ou g = dTVT/dMD.

    Choix de parametrisation, et il est structurant: le reseau predit dTVT/dMD
    et non dS/dMD. Avec une tete initialisee a zero, la sortie vaut alors
    EXACTEMENT la baseline "tenir la derniere TVT" (RMSE 15.91), et non la
    baseline "S constant" (RMSE 107.5). Le reseau part donc du meilleur point
    trivial connu et n'apprend que l'ecart.
    La physique dTVT = -dZ + dS reste presente: dZ/dMD est un canal d'entree.
    La continuite au point PS est exacte par construction.

    g, valid: (B, L). i0: (B,) index du point PS.
    """
    L = g.shape[1]
    ar = torch.arange(L, device=g.device)[None, :]
    after = (ar > i0[:, None]).float() * valid
    return torch.cumsum(g * after * ds, dim=1) * after, after


def reconstruct_direct(u, i0, valid):
    """Variante DIRECTE: le reseau sort delta lui-meme, recale a 0 au point PS.

    Mesure (voir docs): la variante integrale ci-dessus s'effondre sur la
    baseline plate — un biais constant sur g s'integre en une erreur qui croit
    lineairement, donc la sortie sure est zero. En sortant delta directement,
    l'erreur ne s'amplifie pas et le reseau peut exprimer une polyligne
    (la forme reelle de la cible, ~7 segments) sans risque d'integration.
    La continuite au PS reste EXACTE: on soustrait la valeur au point PS.
    """
    L = u.shape[1]
    ar = torch.arange(L, device=u.device)[None, :]
    after = (ar > i0[:, None]).float() * valid
    u0 = torch.gather(u, 1, i0[:, None].clamp(0, L - 1))
    return (u - u0) * after, after
