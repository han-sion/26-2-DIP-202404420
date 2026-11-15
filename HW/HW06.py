# -*- coding: utf-8 -*-
"""
HW06 - Object Counting 웹사이트 (Streamlit)

HW05의 흐름(전처리 -> thresholding -> connected component -> counting)을 기반으로,
강의노트 05(Morphological)와 06(Image Segmentation)의 내용을 적용했습니다.

  - 06: shading 보정, Otsu / Basic Global / Variable thresholding,
        edge 근처 픽셀만 사용한 히스토그램 개선
  - 05: opening, closing, hole filling, boundary extraction, connected component

실행 방법:  streamlit run object_counting.py
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from scipy.cluster.vq import kmeans2
import streamlit as st
from PIL import Image, ImageOps

MAX_SIDE = 800  # 모든 영상의 긴 변을 이 크기로 맞춰서 처리

CH_BG = '배경과의 색 차이'
CH_LIST = [CH_BG, 'Grayscale', 'Saturation (채도)', 'R', 'G', 'B']
TH_AUTO = '자동 (histogram valley)'
TH_OTSU = 'Otsu'
TH_BGT = 'Basic Global Thresholding'
TH_VAR = 'Variable Thresholding'
TH_MANUAL = '직접 입력'
VALLEY_EDGE = 0.4       # valley 바닥에서 이 비율만큼 올라온 지점(왼쪽)을 임계값으로 한다
ETA_MIN = 0.85          # 2단계 Otsu를 적용할 separability 기준
HOLE_FILL_RATIO = 0.8  # 구멍의 평균 밝기가 T의 이 비율보다 크면 물체의 일부로 보고 메운다


# ---------------------------------------------------------------------------
# 1. 영상 불러오기 (HW05 - 1번)
# ---------------------------------------------------------------------------
def load_image(file, max_side=MAX_SIDE):
    """업로드된 파일을 RGB 배열로 읽고, 크기를 맞춘다."""
    I = Image.open(file)
    I = ImageOps.exif_transpose(I)  # 휴대폰 사진의 회전 정보 반영
    I = I.convert('RGB')
    # 영상 크기가 달라도 같은 파라미터가 통하도록 긴 변을 항상 max_side로 맞춘다
    scale = max_side / max(I.size)
    if scale != 1:
        I = I.resize((round(I.size[0] * scale), round(I.size[1] * scale)), Image.LANCZOS)
    return np.array(I)


# ---------------------------------------------------------------------------
# 2. 전처리 (HW05 - 2번: median filter로 noise 줄이기)
# ---------------------------------------------------------------------------
def denoise(rgb, ksize):
    """R, G, B 각 채널에 median filter를 적용한다."""
    f = rgb.astype(float)
    if ksize <= 1:
        return f
    return nd.median_filter(f, size=(ksize, ksize, 1))


def to_single_channel(rgb_f, channel):
    """thresholding에 사용할 1채널 영상을 만든다 (배경과의 색 차이 제외)."""
    if channel == 'Grayscale':
        return rgb_f @ np.array([0.299, 0.587, 0.114])
    if channel == 'Saturation (채도)':
        mx, mn = rgb_f.max(axis=2), rgb_f.min(axis=2)
        return 255 * (mx - mn) / (mx + 1e-9)
    return rgb_f[:, :, ['R', 'G', 'B'].index(channel)]


def fit_background(rgb_f, select, degree):
    """
    select로 지정한 (배경이라고 생각되는) 픽셀들에 매끄러운 곡면을 맞춰서
    영상 전체의 배경 밝기 패턴(shading pattern)을 추정한다.
    degree=1: 평면, degree=2: 2차 곡면
    """
    H, W, _ = rgb_f.shape
    yy, xx = np.mgrid[0:H, 0:W]
    x = (xx / W - 0.5).ravel()
    y = (yy / H - 0.5).ravel()
    cols = [np.ones_like(x), x, y]
    if degree == 2:
        cols += [x * x, x * y, y * y]
    A = np.stack(cols, axis=1)

    idx = np.flatnonzero(select.ravel())
    idx = idx[::max(1, len(idx) // 20000)]  # 속도를 위해 일부 픽셀만 사용
    bg = np.zeros_like(rgb_f)
    for c in range(3):
        v = rgb_f[:, :, c].ravel()
        coef = np.linalg.lstsq(A[idx], v[idx], rcond=None)[0]
        # 배경이 아닌 픽셀(잔차가 큰 픽셀)을 빼고 한 번 더 맞춘다
        res = v[idx] - A[idx] @ coef
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        ok = np.abs(res) < 2.5 * s
        if ok.sum() > 100:
            coef = np.linalg.lstsq(A[idx][ok], v[idx][ok], rcond=None)[0]
        bg[:, :, c] = (A @ coef).reshape(H, W)
    return bg


def border_band(shape, frac=0.04):
    """영상 테두리 띠 (배경이라고 가정하는 영역)"""
    H, W = shape
    w = max(2, int(min(H, W) * frac))
    band = np.zeros((H, W), dtype=bool)
    band[:w, :] = band[-w:, :] = band[:, :w] = band[:, -w:] = True
    return band


def background_difference(rgb_f, select=None, degree=1, return_bg=False):
    """
    배경과의 색 차이 영상.
    배경 패턴을 추정해서 빼고, RGB 공간에서의 거리를 구한다.
    물체가 배경보다 밝든 어둡든, 색만 다르든 모두 큰 값이 된다.
    """
    if select is None:
        select = border_band(rgb_f.shape[:2])
    bg = fit_background(rgb_f, select, degree)
    d = np.clip(np.sqrt(((rgb_f - bg) ** 2).sum(axis=2)), 0, 255)
    return (d, bg) if return_bg else d


def correct_shading(f, sigma):
    """크게 blur한 영상(배경 밝기 패턴)을 빼서 조명 불균일을 보정한다."""
    g = f - nd.gaussian_filter(f, sigma)
    return (g - g.min()) / (g.max() - g.min() + 1e-9) * 255


# ---------------------------------------------------------------------------
# 3. Thresholding (강의노트 06)
# ---------------------------------------------------------------------------
def basic_global_threshold(values, eps=0.5, max_iter=100):
    """
    Basic Global Thresholding (Lab05 Exercise 2)
    1. 초기 임계값 T를 정한다 (평균).
    2. T로 G1(>T), G2(<=T) 두 그룹으로 나눈다.
    3. 각 그룹의 평균 m1, m2를 구한다.
    4. T = (m1 + m2) / 2 로 갱신한다.
    5. T의 변화량 dT가 eps보다 작아질 때까지 2~4를 반복한다.
    """
    T = [float(values.mean())]
    dT = 1000
    n = 0
    while dT > eps and n < max_iter:
        g1 = values[values > T[-1]]
        g2 = values[values <= T[-1]]
        if g1.size == 0 or g2.size == 0:  # 한쪽 그룹이 비면 더 진행할 수 없음
            break
        T.append((g1.mean() + g2.mean()) / 2)
        dT = abs(T[-1] - T[-2])
        n += 1
    return T[-1]


def otsu_threshold(values, return_eta=False):
    """Otsu's method: between-class variance가 최대가 되는 T를 찾는다."""
    hist, edges = np.histogram(values, bins=256, range=(0, 255))
    p = hist / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    P1 = np.cumsum(p)             # class 1의 확률
    m = np.cumsum(p * centers)    # 누적 평균
    mG = m[-1]                    # 전체 평균
    with np.errstate(divide='ignore', invalid='ignore'):
        var_b = (mG * P1 - m) ** 2 / (P1 * (1 - P1))
    var_b = np.nan_to_num(var_b)
    T = float(centers[np.argmax(var_b)])
    if return_eta:  # separability measure: eta = sigma_B^2 / sigma_G^2
        var_g = (p * (centers - mG) ** 2).sum()
        return T, float(var_b.max() / (var_g + 1e-9))
    return T


def two_stage_otsu(values, eta_min):
    """
    밝은 물체와 어두운 물체가 섞여 있으면 Otsu가 '밝은 물체 / 나머지'로 나눠서
    어두운 물체를 배경으로 버리게 된다. 그래서 T 아래쪽 픽셀들만으로 Otsu를
    한 번 더 적용하고, 그 안에서도 두 그룹이 뚜렷이 나뉘면(eta가 크면) 낮은 T를 쓴다.
    """
    T1 = otsu_threshold(values)
    low = values[values <= T1]
    if low.size < 100:
        return T1
    T2, eta = otsu_threshold(low, return_eta=True)
    return T2 if eta >= eta_min else T1


def edge_pixels(f, percentile):
    """
    gradient가 큰 (= 물체와 배경의 경계 근처) 픽셀만 고른다.
    물체가 배경에 비해 아주 작으면 히스토그램에서 물체의 봉우리가 묻히는데,
    경계 근처에서는 물체/배경 픽셀이 비슷한 수로 섞여 있어 valley가 뚜렷해진다.
    """
    g = np.hypot(nd.sobel(f, axis=0), nd.sobel(f, axis=1))
    sel = g >= np.percentile(g, percentile)
    return nd.binary_dilation(sel, iterations=2)


def find_threshold(f, method, use_edge, edge_percentile, T_manual, two_stage=True, eta_min=ETA_MIN):
    if method == TH_MANUAL:
        return float(T_manual)
    if method == TH_AUTO:
        return auto_threshold(f)
    values = f[edge_pixels(f, edge_percentile)] if use_edge else f.ravel()
    if values.size < 100:
        values = f.ravel()
    if method == TH_BGT:
        return basic_global_threshold(values)
    if two_stage:
        return two_stage_otsu(values, eta_min)
    return otsu_threshold(values)


def background_level(f):
    """
    '배경과의 색 차이' 영상에서 배경 픽셀은 0 근처에 몰려 있다.
    테두리 띠(배경이라고 가정)의 값으로 배경이 가질 수 있는 최대 수준을 추정한다:
    median + 4 * (MAD로 구한 표준편차)
    """
    v = f[border_band(f.shape)]
    med = np.median(v)
    return float(med + 4 * 1.4826 * np.median(np.abs(v - med)) + 2)


def valley_threshold(f, min_prominence=0.2):
    """
    히스토그램의 valley(골짜기)로 임계값을 정한다.
    '배경과의 색 차이' 영상에서는 배경이 0 근처의 가장 큰 봉우리이고,
    물체는 그 오른쪽에 또 하나의 봉우리를 만든다. 그 사이의 골짜기가 임계값이다.
    물체 픽셀이 배경보다 훨씬 적어도 보이도록 log 히스토그램을 사용한다.
    뚜렷한 골짜기가 없으면 None을 반환한다.
    """
    hist, _ = np.histogram(f, bins=256, range=(0, 256))
    L = nd.gaussian_filter1d(np.log10(hist + 1.0), 3, mode='nearest')
    p0 = int(np.argmax(L))                      # 배경 봉우리
    vmin, iv = L[p0], p0                        # p0 오른쪽에서 지금까지의 최솟값과 그 위치
    for i in range(p0 + 1, 255):
        if L[i] < vmin:
            vmin, iv = L[i], i
        is_peak = L[i] >= L[i - 1] and L[i] > L[i + 1]
        if is_peak and L[i] - vmin >= min_prominence:
            # 골짜기의 왼쪽 끝 (배경 봉우리가 끝나는 지점)을 임계값으로 한다
            level = vmin + VALLEY_EDGE * (L[i] - vmin)
            left = p0 + int(np.argmax(L[p0:iv + 1] <= level))
            return float(left)
    return None


def auto_threshold(f):
    """
    1순위: 히스토그램에 배경 / 물체 사이의 뚜렷한 valley가 있으면 그 위치.
    2순위: 없으면 Otsu와 배경 수준의 기하평균 T = sqrt(T_배경 * T_otsu).
           (Otsu만 쓰면 물체 밝기가 제각각일 때 배경과 비슷한 색의 물체를 놓치고,
            배경 수준만 쓰면 그림자까지 잡히므로 절충한다)
    """
    T = valley_threshold(f)
    if T is not None:
        return T
    T_otsu = otsu_threshold(f.ravel())
    T_bg = min(background_level(f), T_otsu)
    return float(np.sqrt(T_bg * T_otsu))


def variable_threshold(f, size, a, b):
    """Variable thresholding: T_xy = a * sigma_xy + b * m_xy (local 평균/표준편차)"""
    m = nd.uniform_filter(f, size)
    s = np.sqrt(np.maximum(nd.uniform_filter(f * f, size) - m * m, 0))
    return f > a * s + b * m


def is_object_bright(f, T):
    """테두리를 배경으로 보고, 테두리의 절반 이상이 T보다 밝으면 물체는 어두운 쪽."""
    return (f[border_band(f.shape)] > T).mean() <= 0.5


# ---------------------------------------------------------------------------
# 4. Mask 정리 (강의노트 05: opening, closing, hole filling)
# ---------------------------------------------------------------------------
def color_refine(mask, rgb_f, sat_min=0.2, min_colorful=0.5, chord_min=0.4, step=2):
    """
    알록달록한 물체(레고, 사탕 등)를 위한 처리. mask 안의 픽셀 대부분이 선명한 색일 때만 적용한다.
      (1) 그림자 제외: 채도(saturation)가 낮은 픽셀은 물체가 아니라 그림자로 보고 뺀다.
      (2) 색 경계에서 자르기: 서로 붙어 있어도 색(hue)이 다르면 다른 물체이므로,
          hue가 급격히 바뀌는 경계선을 mask에서 지워서 떼어 놓는다.
    회색/흰색/검은색 물체(동전, 키보드 등)가 있는 영상에서는 아무것도 하지 않는다.
    """
    if mask.sum() == 0:
        return mask
    mx, mn = rgb_f.max(axis=2), rgb_f.min(axis=2)
    sat = (mx - mn) / (mx + 1e-9)
    colorful = (sat > sat_min) & (mx - mn > 40)   # 어두운 픽셀은 noise만으로도 채도가 커지므로 차이값도 확인
    if (colorful & mask).sum() < min_colorful * mask.sum():
        return mask

    mask = mask & colorful                                   # (1) 그림자 제외

    # (2) hue를 단위벡터 (cos, sin)으로 바꾸고, 양옆/위아래 픽셀과의 차이를 본다
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    hue = np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)
    u, v = np.cos(hue), np.sin(hue)
    cut = np.zeros(mask.shape, dtype=bool)
    for axis in (0, 1):
        a = [np.roll(x, step, axis=axis) for x in (u, v, mask)]
        c = [np.roll(x, -step, axis=axis) for x in (u, v, mask)]
        chord = np.hypot(a[0] - c[0], a[1] - c[1])          # hue 차이가 theta면 2*sin(theta/2)
        cut |= (chord > chord_min) & a[2] & c[2]
    return mask & ~cut


def thin_gap_pixels(f_fine, T, gap_r=2, rel=0.4):
    """
    '얇은 골짜기' 픽셀 찾기. grayscale closing은 폭 (2*gap_r)픽셀 이하의 얇은 골짜기를
    양옆 높이만큼 메운다. closing 결과보다 rel 비율 이상 낮은 픽셀 = 얇은 틈.
    (단순히 임계값 아래인 픽셀을 보면, 블러 때문에 얕아진 틈은 놓친다)
    물체의 가장자리(계단 모양 경계)는 closing으로 변하지 않으므로 틈으로 잡히지 않는다.
    """
    p = gap_r + 1
    clo = nd.grey_closing(np.pad(f_fine, p, mode='edge'), footprint=disk(gap_r))[p:-p, p:-p]
    return (clo > T) & (f_fine < (1 - rel) * clo)


def thin_line_pixels(rgb_fine, mask, gap_r=2, min_resp=12, k_noise=6, min_len=15):
    """
    키보드의 위/아래 방향키처럼, 두 물체 사이의 선이 배경색이 아니라 물체와 비슷한 색일 때를 위한
    얇은 선 찾기. 밝기(gray)에서 폭 (2*gap_r)픽셀 이하의 얇은 밝은 선(opening의 잔차)과
    얇은 어두운 선(closing의 잔차)을 찾는다.
    물체 안쪽의 잡음 수준(응답의 90 percentile)의 k_noise배보다 강한 것만 선으로 본다.
    (질감이 거친 물체에서는 기준이 저절로 높아져서 거의 아무것도 안 잡힌다)
    """
    gray = rgb_fine @ np.array([0.299, 0.587, 0.114])
    fp = disk(gap_r)
    p = gap_r + 1
    g = np.pad(gray, p, mode='edge')
    resp = np.maximum(gray - nd.grey_opening(g, footprint=fp)[p:-p, p:-p],
                      nd.grey_closing(g, footprint=fp)[p:-p, p:-p] - gray)
    inside = nd.binary_erosion(mask, iterations=4)
    if inside.sum() < 100:
        return np.zeros_like(mask)
    thr = max(min_resp, k_noise * np.percentile(resp[inside], 90))
    line = (resp > thr) & mask
    # 글자/기호(▲, 알파벳 등)는 짧은 덩어리이므로 버리고, 긴 선(min_len픽셀 이상)만 남긴다
    lab, n = nd.label(line, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, dtype=bool)
    for k, sl in enumerate(nd.find_objects(lab), start=1):
        keep[k] = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) >= min_len
    return keep[lab]


def cut_thin_gaps(mask, gaps, min_piece=0.2):
    """
    붙어 있는 물체 사이의 '아주 얇은 틈'(키보드 키 사이의 선 등)으로 물체를 가른다.
    median filter는 2~3픽셀짜리 얇은 틈을 지워서 물체들을 한 덩어리로 만들어 버리므로,
    필터를 약하게 적용한 영상에서 틈(gaps)을 따로 찾아 둔다.
      1. 틈을 지웠을 때 물체가 '충분히 큰 조각 2개 이상'(각각 원래 넓이의 min_piece 이상)으로
         나뉠 때만 자른다. (글자/무늬 때문에 생긴 흠집은 큰 조각을 못 나누므로 무시된다)
      2. 한 조각이 다른 조각을 둘러싸고 있으면(동전의 테두리 선처럼 고리 모양) 자르지 않는다.
    """
    # 얇은 선은 물체 가장자리 바로 앞에서 끊기는 경우가 많아서, 1픽셀 넓혀서 가장자리까지 닿게 한다
    gaps = nd.binary_dilation(gaps, structure=np.ones((3, 3))) & mask
    if not gaps.any():
        return mask
    cut = mask & ~gaps
    comp, n = nd.label(mask)
    sub, ns = nd.label(cut)
    if ns == 0:
        return mask
    sidx = np.arange(1, ns + 1)
    parent = np.asarray(nd.maximum(comp, sub, index=sidx)).astype(int)
    area_sub = np.bincount(sub.ravel(), minlength=ns + 1)[1:]
    area_comp = np.bincount(comp.ravel(), minlength=n + 1)
    big = (area_sub >= 30) & (area_sub >= min_piece * area_comp[parent])
    count = np.bincount(parent[big], minlength=n + 1)
    adopt = count >= 2
    for k in np.flatnonzero(adopt):          # 고리 모양(조각이 조각을 둘러쌈)이면 취소
        pieces = [i + 1 for i in np.flatnonzero(big & (parent == k))]
        for i in pieces:
            filled = nd.binary_fill_holes(sub == i)
            if any((filled & (sub == j)).sum() > 0.5 * area_sub[j - 1] for j in pieces if j != i):
                adopt[k] = False
                break
    if not adopt.any():
        return mask
    return mask & ~(gaps & adopt[comp])


def interior_objects(mask, f, T, beta=0.0, min_container=0.08, min_feat=4, bg_ratio=0.25, min_area=60, rgb=None):
    """
    '물체 안쪽에 찍힌 물체'를 세는 경우: 검은 가방에 흰 숫자가 찍혀 있을 때처럼,
    영상의 큰 부분을 차지하는 덩어리(컨테이너)가 있고 그 안에 배경이 아닌 구멍이 여러 개 있으면
    덩어리 자체가 아니라 그 구멍들(= 안쪽에 찍힌 물체)을 센다.

    구멍 중에는 '진짜 배경'(손잡이 안쪽의 바닥, 동전 사이의 빈틈)도 있는데, 이것은
    배경과의 색 차이 f가 거의 0이라서 구분된다. 배경이 아닌 구멍(f >= bg_ratio * T)만 센다.
    단, 동전이 빽빽하게 붙어 있는 영상처럼 덩어리가 사실은 '물체 여러 개가 붙은 것'이면
    (덩어리를 나눴을 때 조각 수가 구멍 수보다 많으면) 평소처럼 처리한다.
    조건에 맞지 않으면 None을 반환한다.
    """
    H, W = mask.shape
    comp, n = nd.label(mask)
    if n == 0:
        return None
    area_c = np.bincount(comp.ravel(), minlength=n + 1)
    big = np.flatnonzero(area_c[1:] >= min_container * H * W) + 1
    if big.size == 0:
        return None
    holes = nd.binary_fill_holes(mask) & ~mask
    hl, nh = nd.label(holes)
    if nh == 0:
        return None
    idx = np.arange(1, nh + 1)
    area_h = np.asarray(nd.sum(holes, hl, idx))
    mean_f = np.asarray(nd.mean(f, hl, idx))
    feat = (mean_f >= bg_ratio * T) & (area_h >= min_area)
    # 색이 다른 구멍(숫자는 모두 비슷한 흰색인데 인형 얼굴처럼 색이 다른 것)은 제외
    if rgb is not None:
        col = np.array([nd.mean(rgb[..., c], hl, idx) for c in range(3)]).T
        med = np.median(col[feat], axis=0) if feat.any() else col.mean(axis=0)
        feat &= np.sqrt(((col - med) ** 2).sum(axis=1)) < 40
    # 작은 구멍은 대표 크기의 20% 이상이어야 하지만, f가 아주 큰 얇은 물체(숫자 1)는 살린다
    if feat.sum() >= 3:
        ref = np.median(area_h[feat])
        thin = (mean_f >= 0.6 * T) & (area_h >= 25) & (area_h < 0.2 * ref)
        if rgb is not None:
            thin &= np.sqrt(((col - med) ** 2).sum(axis=1)) < 70
        feat = (feat & (area_h >= 0.2 * ref)) | thin
    keep = np.zeros(nh + 1, dtype=bool)
    markers = None
    for k in big:   # 컨테이너별로, 그 안에 든 구멍만 센다 (구멍 안의 섬은 다른 덩어리이므로 fill_holes로 구분)
        inside = np.unique(hl[nd.binary_fill_holes(comp == k) & holes])
        inside = inside[inside > 0]
        ok = inside[feat[inside - 1] & (area_h[inside - 1] <= 0.1 * area_c[k])]
        if len(ok) >= min_feat:
            n_pieces = 1
            if beta > 0:   # 덩어리를 나눴을 때 조각이 몇 개 나오는가
                if markers is None:
                    markers, _ = find_markers(mask, beta)
                n_pieces = len(np.setdiff1d(np.unique(markers[comp == k]), [0]))
            if len(ok) > n_pieces:
                keep[ok] = True
    if not keep.any():
        return None
    return nd.binary_fill_holes(keep[hl])      # 숫자 8, 6, 0 등의 안쪽 고리까지 포함


def disk(r):
    """반지름 r인 원형 structuring element"""
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2) <= r ** 2


def fill_object_holes(mask, g, T, min_hole=20):
    """
    Hole filling. 단, 구멍이 '진짜 배경'이면 메우지 않는다.
    물체들이 빽빽하게 붙어 있으면 물체 사이의 빈틈도 구멍(hole)이 되는데,
    이것까지 메우면 전체가 한 덩어리가 되어 분리할 수 없기 때문이다.
    g: 물체가 밝게 나오는 영상, T: 임계값
    -> 구멍 안의 평균 밝기가 HOLE_FILL_RATIO * T보다 크면 (물체에 가까우면) 메운다.
    """
    holes = nd.binary_fill_holes(mask) & ~mask
    lab, n = nd.label(holes)
    if n == 0:
        return mask
    idx = np.arange(1, n + 1)
    mean_g = nd.mean(g, lab, index=idx)
    area = nd.sum(holes, lab, index=idx)
    fill = (mean_g > HOLE_FILL_RATIO * T) | (area < min_hole)
    return mask | np.concatenate([[False], fill])[lab]


def clean_mask(mask, open_r=2, close_r=2, fill_holes=True, g=None, T=None):
    if open_r > 0:   # opening: SE보다 작은 점, 가느다란 연결을 제거
        mask = nd.binary_opening(mask, structure=disk(open_r))
    if close_r > 0:  # closing: 좁은 틈과 작은 구멍을 메움 (한 물체가 쪼개지는 것 방지)
        p = close_r + 1
        mask = nd.binary_closing(np.pad(mask, p), structure=disk(close_r))[p:-p, p:-p]
    if fill_holes:   # hole filling
        if g is None:
            mask = nd.binary_fill_holes(mask)
        else:
            mask = fill_object_holes(mask, g, T)
    return mask


# ---------------------------------------------------------------------------
# 5. Connected component 추출 + counting (HW05 - 4번)
# ---------------------------------------------------------------------------
def reconstruct(seed, limit, max_iter=600):
    """
    Morphological reconstruction (강의노트 05의 conditional dilation을 grayscale로):
    X_k = min( dilation(X_{k-1}), limit ) 를 변화가 없을 때까지 반복한다.
    """
    x = np.minimum(seed, limit)
    for i in range(max_iter):
        nx = np.minimum(nd.grey_dilation(x, size=(3, 3)), limit)
        if i % 10 == 9 and np.array_equal(nx, x):
            break
        x = nx
    return x


def find_markers(mask, beta, h0=2.0):
    """distance transform의 봉우리(= 각 물체의 중심)를 marker로 찾는다. (설명은 split_touching 참고)"""
    dist = nd.gaussian_filter(nd.distance_transform_edt(mask), 1.0)
    # 속도를 위해 절반 해상도에서 marker를 찾는다
    d = dist[::2, ::2] / 2
    # 긴 직사각형(스페이스바, 면봉 막대 등)은 능선이 평평해서 높이가 '똑같은' 봉우리가
    # 여러 곳에 생기고, 이것들이 각각 marker로 잡혀 물체 하나가 둘로 갈라진다.
    # 동률일 때는 래스터 순서상 마지막 하나만 이기도록 아주 작은 값을 더해서 깬다.
    d = d + 1e-3 * np.arange(d.size).reshape(d.shape) / d.size
    drop = beta * d + h0 / 2
    rec = reconstruct(d - drop, d)
    peaks = ((d - rec) >= drop - 1e-9) & (d > 0)
    mk_small, _ = nd.label(peaks, structure=np.ones((3, 3)))
    markers = np.repeat(np.repeat(mk_small, 2, axis=0), 2, axis=1)[:mask.shape[0], :mask.shape[1]]
    return markers * mask, dist


def fill_ring_holes(mask, beta):
    """
    가운데 구멍이 뚫린 물체(예: 구멍 있는 동전) 처리.
    고리 모양 물체는 '가장 두꺼운 곳'이 얇아서 자기 marker를 못 받고 옆 물체에 흡수된다.
    그래서 물체에 뚫린 구멍만 골라서 메워준다. 물체 사이의 빈틈과 구별하는 방법:

      각 물체를 (marker 위치 = 중심, 그 점의 distance 값 = 반지름)인 원으로 보고,
      구멍 중심에서 가장 가까운 원의 가장자리까지의 거리 gap을 구한다.
        - 물체 사이의 빈틈: 빈틈 바로 옆이 물체 가장자리 -> gap이 빈틈 크기와 거의 같다
        - 물체에 뚫린 구멍: 자기 물체는 marker가 없으므로 가장 가까운 원은 '옆 물체'이고,
          거기까지는 자기 몸통을 지나가야 한다 -> gap이 구멍 크기보다 훨씬 크다
      => 구멍의 반지름이 gap의 절반보다 작으면 물체에 뚫린 구멍으로 보고 메운다.
    """
    holes = nd.binary_fill_holes(mask) & ~mask
    lab, n = nd.label(holes)
    if n == 0:
        return mask
    markers, dist = find_markers(mask, beta)
    nm = markers.max()
    if nm == 0:
        return mask
    midx = np.arange(1, nm + 1)
    centers = np.array(nd.maximum_position(dist, markers, index=midx), dtype=float)  # (nm, 2)
    radii = np.asarray(nd.maximum(dist, markers, index=midx))

    idx = np.arange(1, n + 1)
    hole_c = np.array(nd.center_of_mass(holes, lab, index=idx))                      # (n, 2)
    hole_r = np.asarray(nd.maximum(nd.distance_transform_edt(holes), lab, index=idx))
    d = np.sqrt(((hole_c[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))        # (n, nm)
    gap = (d - radii[None, :]).min(axis=1)
    fill = np.concatenate([[False], hole_r < 0.5 * gap])
    return mask | fill[lab]


def merge_bridged(labels, dist, k=2.5):
    """
    면봉, 아령처럼 '두꺼운 양 끝이 가늘고 긴 막대로 이어진' 물체가 둘로 나뉘는 것을 막는다.

    붙어 있는 두 물체라면 중심 사이의 거리가 두 반지름의 합(R1 + R2) 정도이다.
    그런데 나뉜 두 조각의 중심 거리가 k * (R1 + R2)보다 훨씬 멀면,
    그 사이를 긴 막대가 잇고 있다는 뜻이므로 하나의 물체로 다시 합친다.

    막대로 이어진 조각이 3개 이상이면 (면봉 여러 개가 겹친 경우) 몇 개의 물체인지
    모호하므로, 조각 2개짜리 물체들의 넓이를 기준으로 개수를 추정한다.
    """
    n = labels.max()
    if n < 2:
        return labels
    idx = np.arange(1, n + 1)
    R = np.asarray(nd.maximum(dist, labels, index=idx))
    C = np.array(nd.maximum_position(dist, labels, index=idx), dtype=float)
    area = np.bincount(labels.ravel(), minlength=n + 1)[1:]

    # 서로 맞닿아 있는 조각 쌍 찾기
    pairs = set()
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        pairs.update(zip(np.minimum(a[m], b[m]).tolist(), np.maximum(a[m], b[m]).tolist()))

    parent = list(range(n + 1))                 # union-find

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged = False
    for i, j in pairs:
        d = np.hypot(*(C[i - 1] - C[j - 1]))
        if d > k * (R[i - 1] + R[j - 1]):       # 긴 막대로 이어져 있음 -> 같은 물체
            parent[find(i)] = find(j)
            merged = True
    if not merged:
        return labels

    groups = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    two = [area[np.array(g) - 1].sum() for g in groups.values() if len(g) == 2]
    unit = np.median(two) if two else None      # 조각 2개짜리 물체 하나의 넓이

    new_id = np.zeros(n + 1, dtype=int)
    count = 0
    for g in groups.values():
        g = np.array(g)
        k_obj = 1
        if len(g) >= 3:                          # 여러 개가 겹친 덩어리: 넓이로 개수 추정
            est = area[g - 1].sum() / unit if unit else len(g) / 2
            k_obj = int(np.clip(round(est), 1, len(g) - 1))
        if k_obj == 1:
            count += 1
            new_id[g] = count
        else:                                    # 화면 표시용으로 조각들을 k_obj개 묶음으로 나눈다
            _, cl = kmeans2(C[g - 1], k_obj, minit='++', seed=0)
            for c in np.unique(cl):
                count += 1
                new_id[g[cl == c]] = count
    return new_id[labels]


def split_touching(mask, beta, h0=2.0, erode_r=0, raw=None):
    """
    서로 붙어 있는 물체 분리.
    distance transform D는 'erosion을 몇 번 하면 그 픽셀이 사라지는가'와 같다.
    물체가 붙어 있으면 D에서 물체 중심마다 봉우리가 생기고, 붙은 부분(neck)은 골짜기가 된다.
    봉우리 높이의 beta배(+h0) 이상 깊은 골짜기로 나뉜 봉우리만 독립된 물체로 인정한다.
      1. seed = D - (beta * D + h0)
      2. seed를 D 아래에서 reconstruction -> 얕은 봉우리는 옆의 큰 봉우리에 묻힌다
      3. 묻히지 않고 남은 봉우리 = 각 물체의 marker
      4. 각 픽셀을 같은 덩어리 안의 가장 가까운 marker에 배정
    """
    comp, ncomp = nd.label(mask)
    if ncomp == 0:
        return comp
    if erode_r > 0:
        # [단순 방식] erosion으로 붙은 부분을 끊고, 남은 조각에 번호를 매긴다 (numbering).
        # 아래에서 각 픽셀을 가장 가까운 조각에 배정해 원래 크기로 복원한다.
        # erosion으로 완전히 사라지는 작은 덩어리는 그대로 물체 1개로 센다.
        # erosion은 thresholding 직후의 mask(raw)에 바로 적용한다 (opening / hole filling 이전).
        eroded = nd.binary_erosion(mask if raw is None else raw, structure=disk(erode_r))
        markers, nmark = nd.label(eroded)
        if nmark > 1:
            # 한 물체가 erosion으로 깨지면서 생긴 부스러기 조각은 번호를 주지 않는다:
            # 조각 크기 중앙값의 15%보다 작은 조각은 버린다.
            a = nd.sum(eroded, markers, index=np.arange(1, nmark + 1))
            keep = np.concatenate([[False], a >= 0.15 * np.median(a)])
            markers = markers * keep[markers]
        markers = markers * mask
    else:
        markers, dist = find_markers(mask, beta, h0)

    labels = np.zeros(mask.shape, dtype=int)
    count = 0
    for k, sl in enumerate(nd.find_objects(comp), start=1):
        sub = comp[sl] == k
        sub_markers = markers[sl] * sub
        ids = np.unique(sub_markers[sub_markers > 0])
        if len(ids) <= 1:  # marker가 하나면 나눌 필요 없음
            count += 1
            labels[sl][sub] = count
            continue
        iy, ix = nd.distance_transform_edt(sub_markers == 0, return_distances=False,
                                           return_indices=True)
        nearest = np.searchsorted(ids, sub_markers[iy, ix])
        labels[sl][sub] = count + 1 + nearest[sub]
        count += len(ids)
    if erode_r == 0:
        labels = merge_bridged(labels, dist)
    return labels


def count_objects(mask, min_ratio=0.1, min_area=30, remove_border=False, split_beta=0.0, erode_r=0, raw=None,
                  drop_lines=True, drop_border_bg=True):
    """
    labeling 후 너무 작은 component(noise)를 제거하고 개수를 센다.
    - min_area  : 이 픽셀 수보다 작으면 제거
    - min_ratio : '대표 물체 크기'의 이 비율보다 작으면 제거
    반환값: (labels, areas)  -> labels는 1..N, areas[i-1]은 i번 물체의 픽셀 수
    """
    if erode_r > 0 or split_beta > 0:
        labels = split_touching(mask, split_beta, erode_r=erode_r, raw=raw)
    else:
        labels, _ = nd.label(mask)
    nums = labels.max()
    if nums == 0:
        return labels, np.array([])

    areas = nd.sum(mask, labels, index=np.arange(1, nums + 1))  # 각 label의 픽셀 수

    def weighted_median(values, weights):
        """가중 중앙값: 전체 weight의 절반이 이 값 이하에 속한다."""
        o = np.argsort(values)
        c = np.cumsum(weights[o])
        return values[o][np.searchsorted(c, c[-1] / 2)]

    idx = np.arange(1, nums + 1)
    ok = areas >= min_area

    # (1) 선 모양 제외: 보도블록 틈, 바닥의 금, 그림자 띠 같은 것.
    #     두께(= distance transform의 최댓값)에 비해 면적이 너무 크면 선 모양이다.
    #     compactness = pi * 두께^2 / 면적  (원이면 1, 가늘고 길수록 0에 가까움)
    #     단, 덩어리 모양 물체들보다 확실히 가는 경우에만 제외한다
    #     (연필처럼 물체 자체가 길쭉한 영상에서는 아무것도 제외하지 않기 위해).
    if drop_lines:
        thick = nd.maximum(nd.distance_transform_edt(mask), labels, index=idx)
        line_like = np.pi * thick ** 2 / areas < 0.15
        blob = ok & ~line_like
        if blob.any() and line_like.any():
            ref = weighted_median(thick[blob], areas[blob])
            ok &= ~(line_like & (thick < 0.5 * ref))

    # (2) 배경 덩어리 제외: 영상 테두리에 길게 닿아 있는 영역은 물체가 아니라
    #     그림자나 배경의 일부일 가능성이 높다. (살짝 잘린 물체는 닿는 길이가 짧다)
    if drop_border_bg:
        H, W = labels.shape
        edge = np.concatenate([labels[0, :], labels[-1, :], labels[1:-1, 0], labels[1:-1, -1]])
        contact = np.bincount(edge, minlength=nums + 1)[1:]
        ok &= contact <= 0.25 * min(H, W)

    # (3) 너무 작은 것 제외. 대표 물체 크기 = 면적 가중 중앙값
    #     (작은 noise가 많아도, 몇 개가 뭉친 큰 덩어리가 있어도 영향을 덜 받는다)
    keep = ok.copy()
    if ok.any():
        typical = weighted_median(areas[ok], areas[ok])
        keep &= areas >= min_ratio * typical

    if remove_border:  # 영상 테두리에 걸친(잘린) 물체 제외
        edge = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
        keep[edge[edge > 0] - 1] = False

    # 남은 물체만 1..N으로 다시 번호 매기기
    new_index = np.zeros(nums + 1, dtype=int)
    new_index[1:][keep] = np.arange(1, keep.sum() + 1)
    return new_index[labels], areas[keep]


def boundary(labels):
    """Boundary extraction: beta(A) = A - (A erosion B), 각 물체에 대해 한 번에 계산"""
    se = np.ones((3, 3))
    inner = (nd.grey_erosion(labels, footprint=se) == labels) & \
            (nd.grey_dilation(labels, footprint=se) == labels)
    return (labels > 0) & ~inner


# ---------------------------------------------------------------------------
# 전체 파이프라인
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_pipeline(rgb, channel=CH_BG, ksize=5, shading=False, shading_sigma=50,
                 method=TH_AUTO, use_edge=False, edge_percentile=90, T_manual=100,
                 var_size=101, var_a=1.0, var_b=1.0, polarity='자동',
                 open_r=2, close_r=0, fill_holes=True,
                 min_ratio=0.10, min_area=30, remove_border=False, split_beta=0.2,
                 two_stage=True, erode_r=0, drop_lines=True, drop_border_bg=True,
                 color_split=True, gap_cut=2, interior_mode=True, chroma_aug=True):
    rgb_f = denoise(rgb, ksize)
    sparse = False
    m_eff = TH_OTSU if (method == TH_VAR or (method == TH_AUTO and channel != CH_BG)) else method
    th = lambda img: find_threshold(img, m_eff,
                                    use_edge, edge_percentile, T_manual, two_stage)

    if channel == CH_BG:
        # 1차: 테두리를 배경으로 보고 평면으로 배경 추정 -> 대략적인 물체 영역
        f, bg = background_difference(rgb_f, return_bg=True)
        T = th(f)
        rough = nd.binary_dilation(f > T, iterations=5)
        # 2차: 물체가 아닌 영역 전체로 배경을 다시 추정 (조명 불균일 보정)
        if 0 < rough.mean() < 0.6:
            f, bg = background_difference(rgb_f, select=~rough, degree=2, return_bg=True)
            T = th(f)
        bright = True
        # 물체가 아주 드문 영상(작은 잎 몇 개 등): 히스토그램 valley가 너무 높게 잡혀
        # 작은 물체가 사라지므로, 상위 1% 밝기의 0.8배 이하로 임계값을 낮춘다.
        sparse = method == TH_AUTO and (f > T).mean() < 0.03
        if sparse:
            min_ratio = min(min_ratio, 0.03)
            min_area = min(min_area, 8)
            open_r = 0
        if sparse and (f > T).mean() < 0.01:
            T = min(T, 0.8 * float(np.percentile(f, 99)))
            min_area = min(min_area, 12)
    else:
        f = to_single_channel(rgb_f, channel)
        if shading:
            f = correct_shading(f, shading_sigma)
        T = th(f)
        bright = {'밝은 물체': True, '어두운 물체': False}.get(polarity, is_object_bright(f, T))

    g = f if bright else 255 - f          # g: 물체가 항상 밝게 나오는 영상
    Tg = T if bright else 255 - T
    if method == TH_VAR:
        mask_raw = variable_threshold(g, var_size, var_a, var_b)
        mask = clean_mask(mask_raw, open_r, close_r, fill_holes)
    else:
        mask_raw = g > Tg
        if sparse and channel == CH_BG:
            # hysteresis: 낮은 임계값(0.65T)으로 물체 전체를 잡되, 높은 임계값(T)을 넘는
            # 픽셀이 있는 덩어리만 남긴다 (흐릿한 잎은 살리고 바닥 질감은 버린다).
            lo = g > 0.65 * Tg
            lab_h, n_h = nd.label(lo)
            if n_h:
                mx = nd.maximum(g, lab_h, np.arange(1, n_h + 1))
                mask_raw = np.isin(lab_h, 1 + np.flatnonzero(mx >= Tg))
        if chroma_aug and channel == CH_BG and method == TH_AUTO:
            gc = chroma_difference(rgb_f, bg)
            Tc = auto_threshold(gc)
            if Tc >= 12 and (g > Tg).mean() < 0.10:
                mask_raw = mask_raw | (nd.binary_opening(gc > Tc, structure=disk(1)))
                min_ratio = min(min_ratio, 0.03)   # 크기가 제각각인 잎: 작은 잎을 노이즈로 오해하지 않게
        if gap_cut and channel == CH_BG:
            # median filter를 약하게(3) 적용한 영상에서 얇은 틈을 찾아 물체를 가른다
            rgb_fine = denoise(rgb, min(3, ksize))
            f_fine = np.clip(np.sqrt(((rgb_fine - bg) ** 2).sum(axis=2)), 0, 255)
            gaps = np.zeros_like(mask_raw)
            if gap_cut in (1, 3):
                gaps |= thin_gap_pixels(f_fine, T)
            if gap_cut in (2, 3):
                gaps |= thin_line_pixels(rgb_fine, mask_raw)
            mask_raw = cut_thin_gaps(mask_raw, gaps)
        mask = clean_mask(mask_raw, open_r, close_r, fill_holes, g, Tg)
    if interior_mode:
        inner = interior_objects(mask_raw, g, Tg, split_beta if erode_r == 0 else 0.0, rgb=rgb_f)
        if inner is not None:    # 가방 위의 숫자처럼 '물체 안쪽의 물체'를 세는 영상
            labels, areas = count_objects(inner, 0.0, 20, remove_border,
                                          drop_lines=False, drop_border_bg=False)
            return dict(f=f, T=T, bright=bright, mask_raw=mask_raw, mask=inner, eroded=None,
                        labels=labels, areas=areas)
    if color_split and channel == CH_BG:
        mask = color_refine(mask, rgb_f)
        mask_raw = mask_raw & nd.binary_dilation(mask, iterations=2)
    if fill_holes and erode_r == 0 and split_beta > 0:
        mask = fill_ring_holes(mask, split_beta)

    # erosion 전에 물체 안쪽의 구멍(글자, 무늬 등)을 먼저 메운다.
    # 구멍이 있는 채로 erosion하면 구멍이 커져서 물체 하나가 여러 조각으로 갈라진다.
    seed = mask_raw
    if fill_holes:
        seed = nd.binary_fill_holes(mask_raw) if method == TH_VAR else fill_object_holes(mask_raw, g, Tg)
    labels, areas = count_objects(mask, min_ratio, min_area, remove_border, split_beta, erode_r,
                                  raw=seed, drop_lines=drop_lines, drop_border_bg=drop_border_bg)
    if sparse and channel == CH_BG and len(areas):
        labels, areas = drop_dark_cracks(labels, areas, rgb_f, bg)
    if channel == CH_BG and not sparse and method == TH_AUTO:
        labels, areas = pale_round_objects(labels, areas, f, T)
        labels, areas = merge_fragments(labels, areas, f, T)
    if channel == CH_BG and not sparse and method == TH_AUTO and color_split:
        labels, areas = add_faint_objects(labels, areas, f, T)
    eroded = nd.binary_erosion(seed, structure=disk(erode_r)) if erode_r > 0 else None
    return dict(f=f, T=T, bright=bright, mask_raw=mask_raw, mask=mask, eroded=eroded,
                labels=labels, areas=areas)


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def show_img(img, title=None, cmap='gray'):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    if title:
        ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)


def drop_dark_cracks(labels, areas, rgb_f, bg):
    """
    바닥보다 확실히 어두운 조각 중 작거나 길쭉한 것(균열, 어두운 점)을 제외한다.
    (낙엽은 바닥보다 밝다. 어두운 갈색 낙엽은 덩어리 모양이라 남는다.)
    """
    n = len(areas)
    idx = np.arange(1, n + 1)
    dl = nd.mean(rgb_f.mean(axis=2) - bg.mean(axis=2), labels, idx)   # 바닥 대비 밝기 차이
    thick = nd.maximum(nd.distance_transform_edt(labels > 0), labels, idx)
    sl = nd.find_objects(labels)
    longside = np.array([max(s[0].stop - s[0].start, s[1].stop - s[1].start) for s in sl])
    ratio = 2 * thick / np.maximum(longside, 1)
    bad = (dl < -20) & ((areas < 100) | (ratio < 0.7))
    if not bad.any():
        return labels, areas
    new = np.zeros_like(labels)
    keep = np.flatnonzero(~bad)
    for k, i in enumerate(keep):
        new[labels == i + 1] = k + 1
    return new, areas[keep]


def add_faint_objects(labels, areas, f, T, lo=0.3, sol_min=0.9):
    """
    흰색이나 투명한 물체는 바닥과 색이 비슷해서 임계값을 못 넘는다. 낮은 임계값(0.3T)에서
    '이미 센 물체와 겹치지 않고, 꽉 찬 덩어리이며, 크기가 다른 물체와 비슷한' 것을 추가한다.
    (바닥 무늬는 보통 작거나 들쭉날쭉해서 크기 / 꽉 찬 정도 조건에서 걸러진다)
    """
    n = len(areas)
    if n < 5:
        return labels, areas
    low = nd.binary_opening(f > lo * T)
    fil = nd.binary_fill_holes(low)
    cand, m = nd.label(fil)
    if m == 0:
        return labels, areas
    idx = np.arange(1, m + 1)
    ar = np.asarray(nd.sum(fil, cand, idx))
    ov = np.asarray(nd.sum(labels > 0, cand, idx))
    sol = np.asarray(nd.sum(low, cand, idx)) / np.maximum(ar, 1)
    # 배경이 깨끗한 영상에서만 동작: 물체 주변을 뺀 나머지에서 낮은 임계값을 넘는 비율이 2% 미만
    if (low & ~nd.binary_dilation(labels > 0, iterations=6)).mean() >= 0.02:
        return labels, areas
    med = np.median(areas)
    ok = (ov == 0) & (sol >= sol_min) & (ar >= 0.5 * med) & (ar <= 2 * med)
    # 이미 센 물체(그림자 포함)와 충분히 떨어진, 혼자 있는 후보만 (물체 옆의 그림자 제외)
    near = nd.binary_dilation(labels > 0, iterations=4)
    ok &= np.asarray(nd.sum(near, cand, idx)) == 0
    # 후보 주변 고리의 f가 거의 0이어야 한다 (바닥 무늬가 아닌 독립된 물체)
    ring = nd.binary_dilation(fil, iterations=6) & ~nd.binary_dilation(fil, iterations=2)
    rl = np.where(ring, nd.grey_dilation(cand, size=(15, 15)), 0)
    rm = np.asarray(nd.mean(f, rl, idx))
    ok &= rm < 0.35 * T
    edge = np.zeros_like(cand, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    ok &= np.asarray(nd.sum(edge, cand, idx)) == 0          # 테두리에 닿은 후보 제외
    if not ok.any():
        return labels, areas
    new = labels.copy()
    add = []
    for i in np.flatnonzero(ok):
        new[cand == i + 1] = n + len(add) + 1
        add.append(ar[i])
    return new, np.concatenate([areas, add])


def merge_fragments(labels, areas, f, T, gap=8, small=0.5, lo=0.4, hi=1.4, faint=1.4):
    """
    색이 연한(배경과의 차이가 임계값의 faint배 미만인) 동전은 한 개가 조각 2개로 갈라져 셈해지는 일이 있다. 둘 다 작은 조각이고
    합친 모양의 convex hull이 대표 크기의 물체와 비슷하고 둥글면 하나로 합친다.
    """
    n = len(areas)
    if n < 5:
        return labels, areas
    med = np.median(areas)
    idx = np.arange(1, n + 1)
    smalls = [i for i in idx if areas[i - 1] < small * med]
    if len(smalls) < 2:
        return labels, areas
    new = labels.copy()
    parent = {i: i for i in smalls}
    for a in smalls:
        da = nd.binary_dilation(labels == a, iterations=gap)
        for b in smalls:
            if b <= a or parent[b] != b or parent[a] != a:
                continue
            if not (da & (labels == b)).any():
                continue
            u = (labels == a) | (labels == b)
            sl = nd.find_objects(u.astype(int))[0]
            sub = u[sl]
            hull = convex_hull_mask(sub)
            if hull is None:
                continue
            h = hull.sum()
            circ = 4 * np.pi * h / max(perimeter_len(hull), 1) ** 2
            if f[u].mean() < faint * T and lo * med <= h <= hi * med and circ > 0.85 and sub.sum() >= 0.3 * h:
                full = np.zeros_like(u)
                full[sl] = hull
                new[full & ((labels == 0) | (labels == a) | (labels == b))] = a
                parent[b] = a
    keep = [i for i in idx if parent.get(i, i) == i]
    out = np.zeros_like(new)
    for k, i in enumerate(keep):
        out[new == i] = k + 1
    return out, np.array([(out == k + 1).sum() for k in range(len(keep))], dtype=float)


def convex_hull_mask(m):
    """작은 마스크 m의 convex hull (scipy만 사용)."""
    from scipy.spatial import ConvexHull, Delaunay
    pts = np.argwhere(m)
    if len(pts) < 5:
        return None
    try:
        tri = Delaunay(pts[ConvexHull(pts).vertices])
    except Exception:
        return None
    yy, xx = np.indices(m.shape)
    return tri.find_simplex(np.c_[yy.ravel(), xx.ravel()]).reshape(m.shape) >= 0


def perimeter_len(m):
    return float((m & ~nd.binary_erosion(m)).sum())


def pale_round_objects(labels, areas, f, T, min_ids=3):
    """
    색이 연한 둥근 동전이 다른 동전들 사이에 끼어 있으면, 이웃 동전의 영역에 먹혀서 조각조각 난다.
    '배경과의 차이가 중간 정도(0.4T ~ 2.2T)'인 영역은 연한 동전의 몸통이므로, 이 영역에서
    꽉 차고 둥근 덩어리가 3개 이상의 조각(label)에 걸쳐 있으면 그 덩어리를 한 개의 물체로 만든다.
    """
    n = len(areas)
    if n < 5:
        return labels, areas
    med = np.median(areas)
    band = nd.binary_opening((f > 0.4 * T) & (f < 2.2 * T), structure=disk(5))
    cand, m = nd.label(band)
    new = labels.copy()
    changed = False
    nxt = labels.max() + 1
    for i, sl in enumerate(nd.find_objects(cand), start=1):
        mm = cand[sl] == i
        ar = mm.sum()
        if not (0.5 * med <= ar <= 1.4 * med):
            continue
        hull = convex_hull_mask(mm)
        if hull is None:
            continue
        sol = ar / hull.sum()
        circ = 4 * np.pi * hull.sum() / max(perimeter_len(hull), 1) ** 2
        ids = np.unique(labels[sl][mm])
        ids = ids[ids > 0]
        if sol > 0.88 and circ > 0.9 and len(ids) >= min_ids:
            full = np.zeros(labels.shape, dtype=bool)
            full[sl] = mm
            new[full] = nxt
            nxt += 1
            changed = True
    if not changed:
        return labels, areas
    # 조각이 너무 작아진 label은 버리고 번호를 다시 매긴다
    keep = []
    for k in range(1, nxt):
        a = (new == k).sum()
        if k <= n:
            if a >= 0.5 * (labels == k).sum() and a > 0:
                keep.append(k)
        elif a > 0:
            keep.append(k)
    out = np.zeros_like(new)
    for j, k in enumerate(keep):
        out[new == k] = j + 1
    return out, np.array([(out == j + 1).sum() for j in range(len(keep))], dtype=float)


def chroma_difference(rgb_f, bg):
    """밝기를 빼고 '색 성분'만 배경과 비교한 차이 (회색 바닥 위의 누런 낙엽처럼 색이 있는 물체용)."""
    d = (rgb_f - rgb_f.mean(axis=2, keepdims=True)) - (bg - bg.mean(axis=2, keepdims=True))
    return 1.5 * np.sqrt((d ** 2).sum(axis=2))


def show_overlay(rgb, labels, areas):
    """원본 위에 물체의 경계선과 번호를 표시한다."""
    n = len(areas)
    out = rgb.copy()
    edge = nd.binary_dilation(boundary(labels))
    out[edge] = [255, 230, 0]
    fig, ax = plt.subplots(figsize=(10, 10 * rgb.shape[0] / rgb.shape[1]))
    ax.imshow(out)
    ax.imshow(np.ma.masked_where(labels == 0, labels), cmap='Spectral_r', alpha=0.3)
    if n > 0:
        centers = nd.center_of_mass(labels > 0, labels, index=np.arange(1, n + 1))
        fs = 11 if n <= 60 else 7
        for i, (y, x) in enumerate(centers):
            ax.text(x, y, str(i + 1), color='white', fontsize=fs, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', ec='none', alpha=0.6))
    ax.axis('off')
    fig.tight_layout(pad=0)
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Streamlit 화면
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title='HW06', layout='wide', initial_sidebar_state='collapsed')
    st.title('HW06')
    st.caption('전자공학과 202404420 한시온')
    st.write('이미지를 올리면 **전처리 → Thresholding → Morphology → Connected Component** '
             '순서로 물체의 개수를 셉니다.')

    file = st.file_uploader('이미지 파일을 선택하세요',
                            type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp'])

    # ----- 파라미터 (모두 기본값으로 자동 처리) -----
    channel, ksize = CH_BG, 5
    shading, shading_sigma, polarity = False, 50, '자동'
    method, use_edge, edge_percentile, two_stage, T_manual = TH_AUTO, False, 90, True, 100
    is_var, var_size, var_a, var_b = False, 101, 1.0, 1.0
    open_r, close_r, fill_holes = 2, 0, True
    split_beta, erode_r = 0.2, 0
    min_ratio, min_area = 0.10, 30
    color_split, gap_cut, interior_mode = True, True, True
    drop_lines, drop_border_bg, remove_border = True, True, False

    if file is None:
        st.info('위에서 이미지를 업로드하면 결과가 표시됩니다.')
        return

    rgb = load_image(file)
    with st.spinner('처리 중...'):
        r = run_pipeline(rgb, channel, ksize, shading, shading_sigma,
                         method, use_edge, edge_percentile, T_manual,
                         var_size, var_a, var_b, polarity,
                         open_r, close_r, fill_holes,
                         min_ratio, min_area, remove_border, split_beta, two_stage, erode_r, drop_lines, drop_border_bg,
                         color_split, 2 if gap_cut else 0, interior_mode)
    labels, areas = r['labels'], r['areas']
    n = len(areas)

    # ----- 결과 -----
    st.header(f'결과: 물체 {n}개')
    c1, c2, c3 = st.columns(3)
    c1.metric('Object 개수', n)
    c2.metric('임계값 T', '-' if is_var else f"{r['T']:.1f}")
    c3.metric('물체', '배경과 색이 다른 영역' if channel == CH_BG
              else ('배경보다 밝음' if r['bright'] else '배경보다 어두움'))

    show_overlay(rgb, labels, areas)

    if n == 0:
        st.warning('물체를 찾지 못했습니다.')
        return

    # ----- 처리 과정 -----
    st.header('처리 과정')
    c1, c2, c3 = st.columns(3)
    with c1:
        show_img(r['f'], '1. image for thresholding')
    with c2:
        fig, ax = plt.subplots()
        ax.hist(r['f'].ravel(), bins=256, color='gray')
        ax.set_yscale('log')
        if not is_var:
            ax.axvline(r['T'], color='r', linestyle='--', label=f"T = {r['T']:.1f}")
            ax.legend()
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency (log)')
        ax.set_title('2. histogram & threshold')
        st.pyplot(fig)
        plt.close(fig)
    with c3:
        show_img(r['mask_raw'], '3. binary mask')

    c1, c2, c3 = st.columns(3)
    with c1:
        if r['eroded'] is not None:
            show_img(r['eroded'], '4. hole filling + erosion (numbering is done here)')
        else:
            show_img(boundary(labels), '4. boundary')
    with c2:
        show_img(r['mask'], '5. opening / hole filling (only for restoring)')
    with c3:
        show_img(np.ma.masked_where(labels == 0, labels), '6. numbered objects (restored)',
                 cmap='Spectral_r')

    # ----- 가장 큰 / 작은 물체 (HW05 - 5번) -----
    st.header('가장 큰 물체와 가장 작은 물체')
    ind_max = int(np.argmax(areas))
    ind_min = int(np.argmin(areas))
    c1, c2 = st.columns(2)
    with c1:
        show_img(labels == ind_max + 1,
                 f'the largest: index {ind_max + 1}, {int(areas[ind_max])} pixels', cmap='Spectral_r')
    with c2:
        show_img(labels == ind_min + 1,
                 f'the smallest: index {ind_min + 1}, {int(areas[ind_min])} pixels', cmap='Spectral_r')

    with st.expander('물체별 픽셀 수'):
        st.dataframe({'index': np.arange(1, n + 1), 'number of pixels': areas.astype(int)},
                     hide_index=True)



if __name__ == '__main__':
    main()
