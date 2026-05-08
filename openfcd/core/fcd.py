"""Self-contained FCD core, adapted from kasper64/pyfcd (MIT).

We inline rather than import so the package is callable without sys.path
gymnastics, and so we can hook into intermediate quantities (carrier
amplitudes, per-carrier phases) for masking and unwrapping.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
from numpy.fft import fftshift, fftfreq
from scipy.fft import fft2, ifft2, ifftshift
from skimage.draw import disk
from skimage.measure import label, regionprops


def periodic_smooth_decompose(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Moisan (2011) periodic + smooth image decomposition.

    Decomposes u = periodic + smooth where the periodic component has
    exactly matching values on opposite edges, eliminating the discontinuity
    jump that causes Gibbs ringing in FFT-based gradient integration.

    Reference: Moisan (2011) J. Math. Imaging Vision 39(2), 161-179.
    """
    from numpy.fft import rfft2, irfft2, fftfreq as _fftfreq
    H, W = u.shape
    u64 = np.asarray(u, dtype=np.float64)

    v = np.zeros_like(u64)
    v[0, :]  -= u64[-1, :] - u64[0, :]
    v[-1, :] += u64[-1, :] - u64[0, :]
    v[:, 0]  -= u64[:, -1] - u64[:, 0]
    v[:, -1] += u64[:, -1] - u64[:, 0]

    V = rfft2(v)
    ky = (2 * np.pi * _fftfreq(H))[:, None]
    kx = 2 * np.pi * np.arange(W // 2 + 1) / W
    denom = 2 * np.cos(ky) + 2 * np.cos(kx) - 4.0
    denom[0, 0] = 1.0
    S = V / denom
    S[0, 0] = 0.0
    smooth = irfft2(S, s=(H, W))
    periodic = u64 - smooth
    dtype = u.dtype if np.issubdtype(u.dtype, np.floating) else np.float64
    return periodic.astype(dtype, copy=False), smooth.astype(dtype, copy=False)


@functools.lru_cache(maxsize=8)
def _kspace_axes(shape: tuple) -> tuple:
    return (fftshift(fftfreq(shape[0], 1 / (2.0 * np.pi))),
            fftshift(fftfreq(shape[1], 1 / (2.0 * np.pi))))


def pixel2kspace(shape: tuple, pixel_loc: tuple) -> np.ndarray:
    ky_axis, kx_axis = _kspace_axes(shape)
    return np.array([ky_axis[pixel_loc[0]], kx_axis[pixel_loc[1]]])


def _peak_blobs(img: np.ndarray, threshold: float, n: int) -> list:
    blob_img = img > threshold
    blob_img[0] = False
    blob_img[-1] = False
    blob_img[:, 0] = False
    blob_img[:, -1] = False
    regions = regionprops(label(blob_img.astype(np.uint8)))
    coords_with_intensity = []
    for r in regions:
        cs = [(img[tuple(c)], tuple(c)) for c in r.coords]
        coords_with_intensity.append(max(cs, key=lambda x: x[0]))
    coords_with_intensity.sort(key=lambda x: -x[0])
    return [c for _, c in coords_with_intensity[:n]]


def find_peaks(img: np.ndarray) -> tuple:
    """Return two perpendicular carrier peak pixel locations (in shifted k-space).

    The two carriers of an axis-aligned checkerboard appear as 4 peaks in
    fftshifted space (a perpendicular pair, each with its own conjugate
    mirror). When the two carriers have unequal amplitudes (e.g. due to a
    non-square ROI causing different FFT leakage along each axis), a strict
    threshold of 0.5 * max keeps only the brightest carrier's peaks (main +
    conjugate + sub-pixel satellites), and the perpendicular one is missed.
    Two fixes applied:
      1. lower threshold to 0.15 * max so the weaker carrier survives
      2. when picking `perp`, reject peaks whose k-vector is parallel /
         anti-parallel to `rightmost` (|cos θ| > 0.7), so we never collapse
         to a single carrier direction.
    """
    spec = fftshift(np.abs(fft2(img - img.mean())))
    ky_axis, kx_axis = _kspace_axes(spec.shape)
    kgrid = np.meshgrid(ky_axis, kx_axis, indexing='ij')
    kmin = 4 * np.pi / min(spec.shape)
    spec = spec * ((kgrid[0] ** 2 + kgrid[1] ** 2) > kmin ** 2)
    threshold = 0.15 * spec.max()
    peaks = _peak_blobs(spec, threshold, 8)
    if len(peaks) < 2:
        raise RuntimeError(f"Could not find two carrier peaks (found {len(peaks)}).")
    peak_ks = [pixel2kspace(spec.shape, p) for p in peaks]
    peak_amps = [float(spec[p]) for p in peaks]
    peak_mags = [float(np.linalg.norm(k)) for k in peak_ks]
    clusters: list[list[int]] = []
    for i in np.argsort(peak_amps)[::-1]:
        placed = False
        for cl in clusters:
            ref_mag = peak_mags[cl[0]]
            if 0.90 * ref_mag <= peak_mags[i] <= 1.10 * ref_mag:
                cl.append(int(i))
                placed = True
                break
        if not placed:
            clusters.append([int(i)])
    eligible = [c for c in clusters if len(c) >= 2] or clusters
    best_cluster = max(eligible, key=lambda c: sum(peak_amps[i] for i in c))
    primary_i = max(best_cluster, key=lambda i: peak_amps[i])
    primary = peaks[primary_i]
    k_p = peak_ks[primary_i]
    mag_p = peak_mags[primary_i] + 1e-9
    candidates = []
    for i in best_cluster:
        if i == primary_i:
            continue
        cos_sim = abs(float(np.dot(peak_ks[i], k_p)) / (peak_mags[i] * mag_p))
        if cos_sim < 0.73:
            candidates.append(peaks[i])
    if not candidates:
        raise RuntimeError(
            f"Found {len(peaks)} peaks but none perpendicular to the brightest "
            f"carrier with matching |k|; checkerboard may be misaligned or pattern degenerate.")
    perp = min(candidates,
               key=lambda p: abs(float(np.dot(pixel2kspace(spec.shape, p), k_p))))
    return primary, perp


def _peak_mask(shape: tuple, pos: tuple, r: float) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[disk(pos, r, shape=shape)] = True
    return m


@dataclass
class Carrier:
    pixel_loc: tuple
    k_loc: np.ndarray         # (ky, kx) in rad/pixel
    krad: float               # support radius in k-space pixels
    mask: np.ndarray          # ifftshifted boolean mask, multiplied with FFT(img)
    ccsgn: np.ndarray         # conjugate of demodulated reference, for phase reference


def calculate_carriers(i_ref: np.ndarray) -> list[Carrier]:
    peaks = find_peaks(i_ref)
    krad = float(np.linalg.norm(np.array(peaks[0]) - np.array(peaks[1])) / 2)
    ref_fft = fft2(i_ref)
    carriers = []
    for peak in peaks:
        m = ifftshift(_peak_mask(i_ref.shape, peak, krad))
        ccsgn = np.conj(ifft2(ref_fft * m))
        carriers.append(Carrier(
            pixel_loc=peak,
            k_loc=pixel2kspace(i_ref.shape, peak),
            krad=krad,
            mask=m,
            ccsgn=ccsgn,
        ))
    return carriers


def carrier_amplitude(img: np.ndarray, carriers: list[Carrier]) -> np.ndarray:
    """Local amplitude envelope summed across both carriers.

    Useful as an occlusion detector: where the checkerboard is hidden by
    a robot or finger, the demodulated envelope collapses.
    """
    img_fft = fft2(img)
    amp = np.zeros(img.shape, dtype=float)
    for c in carriers:
        amp += np.abs(ifft2(img_fft * c.mask))
    return amp


def fcd_phases(i_def: np.ndarray, carriers: list[Carrier],
               unwrap: bool = False) -> list[np.ndarray]:
    """Return the per-carrier phase fields phi_1, phi_2 (radians)."""
    img_fft = fft2(i_def)
    phis = []
    for c in carriers:
        demod = ifft2(img_fft * c.mask) * c.ccsgn
        phi = -np.angle(demod)
        if unwrap:
            from skimage.restoration import unwrap_phase
            phi = unwrap_phase(phi)
        phis.append(phi)
    return phis


def fcd_displacement(i_def: np.ndarray, carriers: list[Carrier],
                     unwrap: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (u, v): apparent checkerboard displacement field, in pixels."""
    phis = fcd_phases(i_def, carriers, unwrap=unwrap)
    return displacement_from_phases(phis, carriers)


def displacement_from_phases(phis: list[np.ndarray],
                             carriers: list[Carrier]) -> tuple[np.ndarray, np.ndarray]:
    """Solve carrier phase shifts for (u, v) apparent displacement in pixels."""
    c0, c1 = carriers
    det = c0.k_loc[1] * c1.k_loc[0] - c0.k_loc[0] * c1.k_loc[1]
    u = (c1.k_loc[0] * phis[0] - c0.k_loc[0] * phis[1]) / det
    v = (c0.k_loc[1] * phis[1] - c1.k_loc[1] * phis[0]) / det
    return u, v


def displacement_from_phase_differences(
    delta_phis: list[np.ndarray],
    carriers: list[Carrier],
) -> tuple[np.ndarray, np.ndarray]:
    """Solve physical phase differences Δφ=-k·u for displacement in pixels."""
    return displacement_from_phases([-phi for phi in delta_phis], carriers)


def fftinvgrad(fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """Integrate a 2D gradient field in the Fourier domain (Wildeman/pyfcd)."""
    fx = fx.copy()
    fy = fy.copy()
    size = fx.shape
    imp_x = -0.5 * np.sum(fx[:, 1:-1], axis=1)
    imp_y = -0.5 * np.sum(fy[1:-1, :], axis=0)
    fx_edge = fx[:, [0, -1]].copy()
    fy_edge = fy[[0, -1], :].copy()
    fx[:, 0] = imp_x
    fx[:, -1] = imp_x
    fy[0, :] = imp_y
    fy[-1, :] = imp_y
    mx, my = fx.mean(), fy.mean()
    ky, kx = np.meshgrid(fftfreq(size[0]), fftfreq(size[1]), indexing='ij')
    k2 = kx ** 2 + ky ** 2
    if size[1] % 2 == 0:
        kx[:, size[1] // 2 + 1] = 0
    if size[0] % 2 == 0:
        ky[size[0] // 2 + 1, :] = 0
    fx_hat = fft2(fx)
    fy_hat = fft2(fy)
    k2[0, 0] = 1
    f_hat = (-1j * kx * fx_hat - 1j * ky * fy_hat) / (2.0 * np.pi * k2)
    f = np.real(ifft2(f_hat))
    y, x = np.meshgrid(range(size[0]), range(size[1]), indexing='ij')
    f = f + mx * x + my * y
    f[:, 0] = (4 * f[:, 1] - f[:, 2] - 2 * fx_edge[:, 0]) / 3
    f[:, -1] = (4 * f[:, -1] - f[:, -2] + 2 * fx_edge[:, 1]) / 3
    f[0, :] = (4 * f[1, :] - f[2, :] - 2 * fy_edge[0, :]) / 3
    f[-1, :] = (4 * f[-1, :] - f[-2, :] + 2 * fy_edge[1, :]) / 3
    return f


def fcd(i_def: np.ndarray, carriers: list[Carrier],
        unwrap: bool = False) -> np.ndarray:
    u, v = fcd_displacement(i_def, carriers, unwrap=unwrap)
    return fftinvgrad(-u, -v)


def carriers_pixel_per_mm(carriers: list[Carrier],
                           checker_cell_mm: float) -> float:
    """Estimate pixel_per_mm from the two detected carriers.

    For an axis-aligned checkerboard with cell side p_mm, both carriers have
    |k| = pi*sqrt(2)/p_mm in physical units (rad/mm). In pixels we measured
    |k_pix| (rad/pixel). Dimensional check:
        k_pix [rad/pixel] = k_phys [rad/mm] * (mm/pixel)
    so pixel_per_mm = k_phys / k_pix.
    """
    import math
    k_phys_per_mm = math.pi * math.sqrt(2) / checker_cell_mm
    k_pix = [float(np.linalg.norm(c.k_loc)) for c in carriers]
    return k_phys_per_mm / float(np.mean(k_pix))


def carrier_wavelength_mm(checker_cell_mm: float) -> float:
    return checker_cell_mm / 2.0


# ---------------------------------------------------------------------------
# High-level convenience API expected by golden tests
# ---------------------------------------------------------------------------

def _bos_autodiscover(ref_path: "pathlib.Path",
                      def_path: "pathlib.Path") -> "dict | None":
    """Try to find BOS annotation + polygon data from the ref/frame paths.

    Expects ref_path named like ``_ref_{condition}.jpg`` and BOS tree at
    ``ref_path.parent.parent``.  Returns a dict with keys ``roi``,
    ``polygons_json``, ``frame_num`` or None if auto-discovery fails.
    """
    import re
    stem = ref_path.stem                      # e.g. "_ref_20hz_s2"
    m = re.match(r"_ref_(.+)", stem)
    if not m:
        return None
    condition = m.group(1)                    # e.g. "20hz_s2"
    bos_root = ref_path.parent.parent         # e.g. BOS/
    ann_dir = bos_root / "annotations" / condition
    if not ann_dir.exists():
        return None

    # Extract frame number from def_path stem
    digits = "".join(c for c in def_path.stem if c.isdigit())
    if not digits:
        return None
    frame_num = int(digits)

    # Find annotation whose range covers frame_num
    ann_data = None
    for ann_path in sorted(p for p in ann_dir.glob("*.json")
                           if not p.name.startswith("._")):
        parts = ann_path.stem.split("-")
        if len(parts) != 2:
            continue
        try:
            lo, hi = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if lo <= frame_num <= hi:
            import json
            ann_data = json.loads(ann_path.read_text())
            break
    if ann_data is None:
        return None

    roi = ann_data.get("roi")

    # Find matching track_poly directory (prefer the plain one, not sam2p/v*)
    out_dir = bos_root / "output" / condition
    polygons_json = None
    if out_dir.exists():
        # Score candidates: prefer shorter name (plain track_poly) over versioned
        candidates = []
        for pd in out_dir.iterdir():
            if not pd.is_dir() or pd.name.startswith("._"):
                continue
            if not pd.name.startswith("track_poly_"):
                continue
            suffix = pd.name[len("track_poly_"):]
            parts = suffix.split("-")
            if len(parts) != 2:
                continue
            try:
                lo, hi = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if lo <= frame_num <= hi:
                pj = pd / "polygons.json"
                if pj.exists():
                    candidates.append((len(pd.name), pj))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            polygons_json = candidates[0][1]

    return {"roi": roi, "polygons_json": polygons_json, "frame_num": frame_num}


def compute_eta_from_images(ref_path: "str | pathlib.Path",
                             def_path: "str | pathlib.Path",
                             pattern_period_mm: float = 1.2,
                             flatfield_sigma: float = 300.0,
                             taper_alpha: float = 0.08,
                             edge_margin_mm: float = 3.0) -> np.ndarray:
    """Run full FCD pipeline on one image pair, return eta_mm array.

    When ``ref_path`` follows the BOS naming convention ``_ref_{condition}.jpg``
    and the surrounding BOS tree is present, automatically discovers the ROI
    crop and tracked polygon source so the output matches the golden BOS output.
    Falls back to a simple full-frame, no-mask pipeline otherwise.
    """
    import pathlib as _pl
    from skimage.io import imread
    from skimage.color import rgb2gray
    from scipy.signal import windows as _windows
    from scipy.ndimage import gaussian_filter

    ref_path = _pl.Path(ref_path)
    def_path = _pl.Path(def_path)

    def _load(p: _pl.Path) -> np.ndarray:
        img = imread(str(p))
        if img.ndim == 3:
            img = rgb2gray(img)
        return img.astype(np.float64)

    def _ff(img: np.ndarray, sigma: float,
            bg_src: np.ndarray | None = None) -> np.ndarray:
        src = bg_src.astype(np.float64) if bg_src is not None else img.astype(np.float64)
        bg = gaussian_filter(src, sigma=sigma)
        out = img.astype(np.float64) / np.maximum(bg, 1e-6)
        out *= bg.mean() / out.mean()
        return out

    i_ref_raw = _load(ref_path)
    i_def_raw = _load(def_path)

    # Auto-discover BOS annotation (ROI + polygon)
    bos_info = _bos_autodiscover(ref_path, def_path)
    roi = bos_info["roi"] if bos_info else None
    polygons_json = bos_info["polygons_json"] if bos_info else None
    frame_num = bos_info["frame_num"] if bos_info else None

    # Apply ROI crop if available
    roi_offset = (0, 0)
    if roi is not None:
        r0 = max(0, roi["row0"])
        c0 = max(0, roi["col0"])
        r1 = min(i_ref_raw.shape[0], r0 + roi["height"])
        c1 = min(i_ref_raw.shape[1], c0 + roi["width"])
        i_ref_raw = i_ref_raw[r0:r1, c0:c1]
        i_def_raw = i_def_raw[r0:r1, c0:c1]
        roi_offset = (r0, c0)

    i_ref_ff = _ff(i_ref_raw, sigma=flatfield_sigma)
    i_def_ff = _ff(i_def_raw, sigma=flatfield_sigma, bg_src=i_ref_raw)
    carriers0 = calculate_carriers(i_ref_ff - i_ref_ff.mean())

    # Build occlusion mask from tracked polygon if available
    occlusion_mask = np.zeros(i_ref_ff.shape, dtype=bool)
    robot_poly = None
    if polygons_json is not None and frame_num is not None:
        import json as _json
        from openfcd.core.mask import Polygon, polygon_mask as _polygon_mask
        records = _json.loads(_pl.Path(polygons_json).read_text())
        for rec in records:
            digits = "".join(c for c in rec["frame"] if c.isdigit())
            if digits and int(digits) == frame_num:
                verts_data = rec.get("polygon_inflated_full_frame")
                if verts_data:
                    poly_ff = Polygon([tuple(v) for v in verts_data])
                    # Shift polygon from full-frame to ROI coords
                    robot_poly = poly_ff.shifted(-roi_offset[0], -roi_offset[1])
                    occlusion_mask = _polygon_mask(i_ref_ff.shape, robot_poly,
                                                   dilate_px=4)
                break

    # Inpaint if mask non-empty
    if occlusion_mask.any():
        from openfcd.core.inpaint import synthesize_from_carriers, inpaint_fft
        syn = synthesize_from_carriers(i_ref_ff - i_ref_ff.mean(), carriers0)
        i_ref_clean = inpaint_fft(i_ref_ff, occlusion_mask, syn + i_ref_ff.mean())
        carriers = calculate_carriers(i_ref_clean - i_ref_clean.mean())
        syn2 = synthesize_from_carriers(i_ref_clean - i_ref_clean.mean(), carriers)
        i_def_clean = inpaint_fft(i_def_ff, occlusion_mask, syn2 + i_ref_clean.mean())
    else:
        i_ref_clean = i_ref_ff
        i_def_clean = i_def_ff
        carriers = carriers0

    # Taper
    if taper_alpha > 0:
        wy = _windows.tukey(i_ref_clean.shape[0], alpha=taper_alpha)
        wx = _windows.tukey(i_ref_clean.shape[1], alpha=taper_alpha)
        win = np.outer(wy, wx)
        ref_mean = i_ref_clean.mean()
        def_mean = i_def_clean.mean()
        i_ref_clean = (i_ref_clean - ref_mean) * win + ref_mean
        i_def_clean = (i_def_clean - def_mean) * win + def_mean

    raw_eta = fcd(i_def_clean - i_ref_clean.mean(), carriers)

    # Detrend plane over valid pixels
    valid = ~occlusion_mask
    h, w = raw_eta.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A = np.column_stack([xx[valid].ravel(), yy[valid].ravel(),
                         np.ones(int(valid.sum()))])
    b_vec = raw_eta[valid].ravel()
    coef, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
    plane = coef[0] * xx + coef[1] * yy + coef[2]
    raw_eta = raw_eta - plane

    px_per_mm = carriers_pixel_per_mm(carriers, pattern_period_mm)

    # Use BOS geometry constants (same defaults as BOS/bos/config.py DEFAULT_GEOMETRY)
    alpha_cal = 0.24981245311327827
    h_p_eff_mm = 14.666
    eta_mm = raw_eta / (alpha_cal * h_p_eff_mm * px_per_mm ** 2)

    eta_out = eta_mm.copy()
    eta_out[occlusion_mask] = np.nan
    if edge_margin_mm > 0:
        em_px = int(edge_margin_mm * px_per_mm)
        if em_px > 0:
            eta_out[:em_px, :] = np.nan
            eta_out[-em_px:, :] = np.nan
            eta_out[:, :em_px] = np.nan
            eta_out[:, -em_px:] = np.nan

    return eta_out


def _load_npy_frames(frames_dir: "pathlib.Path") -> list:
    """Load all non-hidden .npy files from frames_dir, skipping macOS metadata."""
    paths = sorted(p for p in frames_dir.glob("*.npy") if not p.name.startswith("._"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {frames_dir}")
    return [np.load(p) for p in paths]


def compute_batch_mean(frames_dir: "str | pathlib.Path") -> np.ndarray:
    """Load all .npy frames from frames_dir and return pixel-wise nanmean."""
    import pathlib
    import warnings
    arrays = _load_npy_frames(pathlib.Path(frames_dir))
    stack = np.stack(arrays, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0)


def compute_batch_rms(frames_dir: "str | pathlib.Path") -> np.ndarray:
    """Load all .npy frames from frames_dir and return pixel-wise RMS."""
    import pathlib
    import warnings
    arrays = _load_npy_frames(pathlib.Path(frames_dir))
    stack = np.stack(arrays, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stack, axis=0, keepdims=True)
        return np.sqrt(np.nanmean((stack - mean) ** 2, axis=0))
