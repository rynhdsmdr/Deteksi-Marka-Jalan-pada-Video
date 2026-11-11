"""Pipeline deteksi garis marka lajur UNIVERSAL (tanpa pengaturan per video).

Alur: kalibrasi titik hilang otomatis (sekali per video) -> ROI trapesium -> per frame:
resize -> grayscale -> Gaussian blur -> Canny (ambang otomatis) AND ROI AND masker terang -> HoughLinesP
-> pemilihan garis kiri/kanan -> fitting -> smoothing -> VideoWriter (.mp4).

Fungsi-fungsi di sini SAMA PERSIS dengan yang ada di notebook UTS_Kelompok06real.ipynb.
Hanya memakai opencv-python dan numpy.

Sumber:
- Ambang Canny otomatis dari persentil besar gradien dengan ambang bawah = 0.4 x ambang atas mengikuti cara
  fungsi edge(I, 'canny') pada MATLAB Image Processing Toolbox (MathWorks, dokumentasi "edge").
- Kalibrasi titik hilang (voting), Canny adaptif, dan masker terang disusun dengan bantuan asisten AI
  (Claude, Anthropic) lalu diuji dan disesuaikan oleh Kelompok 06.
"""
import time
import cv2
import numpy as np

TARGET_MAX_SIDE = 1000          # sisi terpanjang setelah resize (aturan: 800-1200 px)

# SATU set parameter untuk SEMUA video. Tidak ada nilai yang disetel per klip:
# ROI, baris acuan, dan ambang Canny dihitung otomatis dari isi videonya sendiri.
PARAM = dict(
    blur_ksize=5,
    # Canny adaptif: ambang atas = persentil ke-97 besar gradien di dalam ROI, ambang bawah = 0.4 x ambang atas
    canny_persentil=97, canny_rasio_bawah=0.4, canny_min=15,
    # Masker "lebih terang dari sekitarnya" (adaptiveThreshold) untuk membuang tepi tekstur aspal
    terang_blok=31, terang_C=10, terang_lebar=7,
    # HoughLinesP untuk deteksi per frame
    hough_rho=1, hough_theta=np.pi / 180, hough_thr=25, hough_min_len=25, hough_max_gap=60,
    # Kalibrasi titik hilang (sekali per video)
    kal_n_frame=30, kal_y_atas=0.05, kal_hough=(30, 40, 20), kal_tol_px=12,
    vp_y_rentang=(0.0, 0.65),    # titik hilang dicari pada 0-65% tinggi frame (kamera boleh menunduk)
    # ROI trapesium yang dibentuk dari titik hilang (semua dalam proporsi tinggi/lebar frame)
    roi_bawah=0.93, roi_turun=0.05, roi_lebar_atas=0.25, pos_ref=0.60,
    aspal_min=0.35,              # baris dianggap bukan jalan lagi bila < 35% piksel tengahnya mirip aspal
    # Pemilihan garis kiri/kanan
    vp_tol=0.15,                 # segmen harus mengarah ke titik hilang (toleransi, proporsi lebar)
    inv_tegak=0.15,              # |dx/dy| < 0.15 dianggap hampir tegak -> marka tengah (sisi kanan)
    kontras_min=20,              # marka harus lebih terang dari aspal di sekitarnya
    gap_klaster=25, bobot_min=40,
    lebar_lajur_min=120,         # lebar lajur minimal (px) pada baris acuan
    alpha=0.4, hold=6,           # smoothing antar-frame (hanya untuk video)
    toleransi_px=50, margin_px=20,   # kriteria evaluasi
)


def resize_frame(frame, target=TARGET_MAX_SIDE):
    h, w = frame.shape[:2]
    s = target / float(max(h, w))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=interp)


def praproses(raw, P=PARAM):
    # BGR -> resize -> grayscale -> Gaussian blur. Mengembalikan (frame, gray, blur).
    frame = resize_frame(raw)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (P["blur_ksize"], P["blur_ksize"]), 0)
    return frame, gray, blur


def buat_mask_roi(shape, roi):
    h, w = shape[:2]
    pts = np.array([[int(round(x * w)), int(round(y * h))] for x, y in roi], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask, pts


def ambang_canny_otomatis(blur, mask, P=PARAM):
    # Ambang Canny dari sebaran BESAR GRADIEN di dalam ROI (bukan dari kecerahan), sehingga ikut menyesuaikan
    # bila video tajam/buram, terang/gelap, atau resolusinya rendah. Sebagian besar ROI adalah aspal tanpa tepi,
    # jadi hanya ~3% piksel dengan gradien terbesar (persentil 97) yang dianggap tepi kuat.
    # Gradien dihitung dengan beda pusat np.gradient lalu dikali 8 agar skalanya setara Sobel 3x3 yang dipakai Canny.
    gy, gx = np.gradient(blur.astype(np.float32))
    besar = 8 * (np.abs(gx) + np.abs(gy))
    high = max(2 * P["canny_min"], float(np.percentile(besar[mask > 0], P["canny_persentil"])))
    low = max(P["canny_min"], P["canny_rasio_bawah"] * high)
    return int(low), int(high)


def mask_marka_terang(blur, P=PARAM):
    # Piksel yang lebih terang daripada rata-rata lingkungannya sebesar C (adaptive threshold dengan C negatif).
    # Marka putih/kuning lolos; retakan, bekas ban, dan bayangan (lebih gelap) tidak lolos.
    # Lalu diperlebar dengan average blur + threshold > 0 agar tepi di kedua sisi marka ikut masuk.
    m = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, P["terang_blok"], -P["terang_C"])
    k = P["terang_lebar"]
    return cv2.threshold(cv2.blur(m, (k, k)), 0, 255, cv2.THRESH_BINARY)[1]


def segmentasi(blur, kal, canny=None, P=PARAM):
    # Canny (ambang otomatis, atau pasangan tetap `canny` untuk eksperimen) AND mask ROI AND masker terang.
    mask, pts = buat_mask_roi(blur.shape, kal["roi"])
    low, high = ambang_canny_otomatis(blur, mask, P) if canny is None else canny
    edges = cv2.Canny(blur, low, high)
    terang = mask_marka_terang(blur, P)
    roi_edges = cv2.bitwise_and(cv2.bitwise_and(edges, mask), terang)
    return dict(edges=edges, mask=mask, pts=pts, terang=terang, roi_edges=roi_edges, canny=(low, high))


def kontras_marka(gray, x1, y1, x2, y2, dalam=8, jauh=(16, 30)):
    # Seberapa terang punggung marka dibanding aspal di kedua sisinya (rata-rata sepanjang segmen).
    h, w = gray.shape
    nilai = []
    for t in np.linspace(0.1, 0.9, 7):
        y = int(np.clip(y1 + t * (y2 - y1), 1, h - 2))
        x = int(round(x1 + t * (x2 - x1)))
        baris = gray[y].astype(np.float32)
        def potong(a, b):
            a = int(np.clip(x + a, 0, w - 1)); b = int(np.clip(x + b, 0, w))
            return baris[a:b] if b > a else baris[a:a + 1]
        puncak = potong(-dalam, dalam + 1).max()
        kiri, kanan = np.median(potong(-jauh[1], -jauh[0])), np.median(potong(jauh[0], jauh[1]))
        nilai.append(puncak - min(kiri, kanan))
    return float(np.mean(nilai))


def klaster_x(segmen, gap):
    # Kelompokkan segmen yang x-nya pada baris acuan berdekatan (satu kelompok = satu garis fisik).
    segmen = sorted(segmen, key=lambda s: s[0])
    kelompok = []
    for s in segmen:
        if kelompok and s[0] - kelompok[-1][-1][0] < gap:
            kelompok[-1].append(s)
        else:
            kelompok.append([s])
    return kelompok


def pilih_garis(garis, gray, y_ref, vp, P=PARAM):
    # Saring segmen yang mengarah ke titik hilang, pisahkan kiri/kanan, lalu ambil garis TERDEKAT ke kendaraan.
    h, w = gray.shape
    vx, vy = vp
    kiri, kanan = [], []
    if garis is None:
        return [], []
    for x1, y1, x2, y2 in np.reshape(garis, (-1, 4)):
        dy, dx = float(y2 - y1), float(x2 - x1)
        if abs(dy) < 10:                       # buang garis (hampir) horizontal: garis kejut, zebra cross
            continue
        inv = dx / dy
        if abs(x1 + inv * (vy - y1) - vx) > P["vp_tol"] * w:   # perpanjangan segmen harus lewat dekat titik hilang
            continue
        if kontras_marka(gray, x1, y1, x2, y2) < P["kontras_min"]:
            continue
        x_ref = x1 + inv * (y_ref - y1)
        s = (x_ref, x1, y1, x2, y2, float(np.hypot(dx, dy)))
        if inv < -P["inv_tegak"] and x_ref < vx:                                  # miring ke kanan-atas
            kiri.append(s)
        elif inv >= -P["inv_tegak"] and x_ref >= vx - P["inv_tegak"] * (y_ref - vy):  # miring ke kiri-atas / tegak
            kanan.append(s)

    def terbaik(kandidat, sisi):
        kel = [k for k in klaster_x(kandidat, P["gap_klaster"]) if sum(s[5] for s in k) >= P["bobot_min"]]
        if not kel:
            return []
        kel.sort(key=lambda k: np.average([s[0] for s in k], weights=[s[5] for s in k]))
        k = kel[-1] if sisi == "kiri" else kel[0]     # kiri: paling kanan; kanan: paling kiri
        return [s[1:] for s in k]
    return terbaik(kiri, "kiri"), terbaik(kanan, "kanan")


def fit_garis(segmen):
    # Regresi x = a*y + b berbobot panjang segmen. Mengembalikan (a, b) atau None.
    if not segmen:
        return None
    ys, xs, ws = [], [], []
    for x1, y1, x2, y2, pj in segmen:
        ys += [y1, y2]; xs += [x1, x2]; ws += [pj, pj]
    ys, xs = np.array(ys, float), np.array(xs, float)
    if np.ptp(ys) < 8:
        return None
    a, b = np.polyfit(ys, xs, 1, w=np.array(ws))
    return float(a), float(b)


def x_pada_baris(model, y):
    return None if model is None else model[0] * y + model[1]


def konsistensi(model_kiri, model_kanan, seg_kiri, seg_kanan, y_ref, y_atas, y_bawah, P=PARAM):
    # Garis kiri harus berada di kiri garis kanan di seluruh tinggi ROI dan lajur tidak boleh terlalu sempit.
    # Bila dilanggar, sisi dengan bukti (total panjang segmen) lebih kecil dibuang. Mengembalikan (kiri, kanan, menyilang).
    if model_kiri is None or model_kanan is None:
        return model_kiri, model_kanan, False
    lebar = lambda y: x_pada_baris(model_kanan, y) - x_pada_baris(model_kiri, y)
    if min(lebar(y_atas), lebar(y_bawah)) > 0 and lebar(y_ref) >= P["lebar_lajur_min"]:
        return model_kiri, model_kanan, False
    bukti_k, bukti_r = sum(s[4] for s in seg_kiri), sum(s[4] for s in seg_kanan)
    return (model_kiri, None, True) if bukti_k >= bukti_r else (None, model_kanan, True)


def deteksi(roi_edges, gray, kal, P=PARAM):
    # HoughLinesP -> pilih segmen kiri/kanan -> fitting garis -> cek konsistensi geometri.
    h = gray.shape[0]
    y_ref, roi = kal["y_ref"] * h, kal["roi"]
    t0 = time.perf_counter()
    garis = cv2.HoughLinesP(roi_edges, P["hough_rho"], P["hough_theta"], P["hough_thr"],
                            minLineLength=P["hough_min_len"], maxLineGap=P["hough_max_gap"])
    waktu_ms = (time.perf_counter() - t0) * 1000
    kiri, kanan = pilih_garis(garis, gray, y_ref, kal["vp"], P)
    mk, mr = fit_garis(kiri), fit_garis(kanan)
    mk, mr, menyilang = konsistensi(mk, mr, kiri, kanan, y_ref, roi[1][1] * h, roi[0][1] * h, P)
    return dict(garis=garis, seg_kiri=kiri, seg_kanan=kanan, model_kiri=mk, model_kanan=mr, menyilang=menyilang,
                waktu_ms=waktu_ms, n_garis=0 if garis is None else len(garis))


def segmen_kalibrasi(gray, blur, P=PARAM):
    # Segmen HoughLinesP pada pita bawah frame yang kemiringannya wajar untuk marka dan lebih terang dari aspal.
    # Mengembalikan daftar (a, b, panjang, y_teratas) dengan persamaan garis x = a*y + b.
    h, w = gray.shape
    pita = np.zeros_like(gray)
    pita[int(P["kal_y_atas"] * h):int(P["roi_bawah"] * h), :] = 255
    low, high = ambang_canny_otomatis(blur, pita, P)
    edges = cv2.bitwise_and(cv2.Canny(blur, low, high), pita)
    thr, min_len, gap = P["kal_hough"]
    garis = cv2.HoughLinesP(edges, 1, np.pi / 180, thr, minLineLength=min_len, maxLineGap=gap)
    hasil = []
    if garis is None:
        return hasil
    for x1, y1, x2, y2 in garis.reshape(-1, 4):
        dy, dx = float(y2 - y1), float(x2 - x1)
        if abs(dy) < 10:
            continue
        a = dx / dy
        if not (0.25 <= abs(a) <= 3.0):
            continue
        if kontras_marka(gray, x1, y1, x2, y2) < P["kontras_min"]:
            continue
        hasil.append((a, x1 - a * y1, float(np.hypot(dx, dy)), float(min(y1, y2))))
    return hasil


def titik_hilang(garis, h, w, P=PARAM):
    # Voting: setiap titik kandidat (grid 8 px) diberi skor = total panjang segmen kiri yang perpanjangannya
    # lewat dalam kal_tol_px dari titik itu, begitu juga untuk segmen kanan. Skor akhir = sqrt(kiri * kanan),
    # sehingga titik hilang harus didukung OLEH KEDUA SISI jalan sekaligus.
    if not garis:
        return None
    xs = np.arange(0.2 * w, 0.8 * w, 8)
    ys = np.arange(P["vp_y_rentang"][0] * h, P["vp_y_rentang"][1] * h, 8)
    X, Y = np.meshgrid(xs, ys)
    skor_kiri, skor_kanan = np.zeros_like(X), np.zeros_like(X)
    for a, b, panjang, y_teratas in garis:
        # segmen harus mengarah ke titik itu DAN seluruhnya berada di bawahnya (marka jalan ada di bawah horizon)
        kena = (np.abs(a * Y + b - X) < P["kal_tol_px"]) & (Y < y_teratas)
        if a < 0:
            skor_kiri += kena * panjang
        else:
            skor_kanan += kena * panjang
    skor = cv2.GaussianBlur(np.sqrt(skor_kiri * skor_kanan).astype(np.float32), (5, 5), 0)
    if skor.max() <= 0:
        return None
    i, j = np.unravel_index(np.argmax(skor), skor.shape)
    return float(X[i, j]), float(Y[i, j])


def profil_aspal(frame, vp, P=PARAM):
    # Untuk tiap baris: proporsi piksel di tengah frame yang warnanya mirip aspal.
    # Contoh aspal diambil tepat di bawah titik hilang (hampir selalu jalan di depan kendaraan).
    # "Mirip aspal" = kecerahan (V pada HSV) dekat dengan contoh DAN saturasi (S) rendah.
    h, w = frame.shape[:2]
    vx, vy = vp
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (9, 9), 0), cv2.COLOR_BGR2HSV)
    V, S = hsv[..., 2].astype(np.float32), hsv[..., 1].astype(np.float32)
    y0, y1 = int(vy + 0.05 * h), int(vy + 0.12 * h)
    x0, x1 = int(max(0, vx - 0.08 * w)), int(min(w, vx + 0.08 * w))
    contoh_v, contoh_s = V[y0:y1, x0:x1], S[y0:y1, x0:x1]
    m = float(np.median(contoh_v))
    sebaran = max(12.0, 1.5 * float(np.percentile(np.abs(contoh_v - m), 75)))
    aspal = (np.abs(V - m) < 2.5 * sebaran) & (S < float(np.median(contoh_s)) + 40)
    xa, xb = int(max(0, vx - 0.2 * w)), int(min(w, vx + 0.2 * w))
    return aspal[:, xa:xb].mean(axis=1)


def batas_bawah_jalan(profil, vy, h, P=PARAM):
    # Baris pertama di bawah titik hilang yang sudah TIDAK mirip aspal lagi (kap mobil, dasbor/bodi motor).
    # Bila jalan terlihat sampai bawah, dipakai batas bawaan roi_bawah.
    prof = np.convolve(np.median(np.array(profil), axis=0), np.ones(9) / 9, mode="same")
    batas = P["roi_bawah"] * h
    for y in range(int(vy + 0.10 * h), int(P["roi_bawah"] * h)):
        if prof[y] < P["aspal_min"]:
            batas = y - 0.02 * h
            break
    return max(batas, vy + 0.20 * h) / h


def roi_dari_titik_hilang(vp, w, h, P=PARAM, y_bawah=None):
    # Trapesium: sisi atas sedikit di bawah titik hilang, sisi bawah selebar frame di atas kap/bodi kendaraan.
    vx, vy = vp
    y_atas = vy / h + P["roi_turun"]
    y_bawah = P["roi_bawah"] if y_bawah is None else y_bawah
    xa_kiri = max(0.0, vx / w - P["roi_lebar_atas"])
    xa_kanan = min(1.0, vx / w + P["roi_lebar_atas"])
    roi = [(0.0, y_bawah), (xa_kiri, y_atas), (xa_kanan, y_atas), (1.0, y_bawah)]
    y_ref = y_atas + P["pos_ref"] * (y_bawah - y_atas)
    return dict(vp=(vx, vy), roi=roi, y_ref=y_ref)


def kalibrasi_video(path, P=PARAM):
    # Sekali per video: ambil kal_n_frame frame yang tersebar merata, cari titik hilang dari segmen marka,
    # lalu cari batas bawah jalan dari warna aspal. Hasilnya ROI trapesium + baris acuan untuk SEMUA frame.
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Video tidak bisa dibuka: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    garis, frames = [], []
    for fi in np.linspace(0, max(n - 1, 0), P["kal_n_frame"]).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, raw = cap.read()
        if not ok:
            continue
        frame, gray, blur = praproses(raw, P)
        frames.append(frame)
        garis += segmen_kalibrasi(gray, blur, P)
    cap.release()
    if not frames:
        raise ValueError(f"Tidak ada frame yang bisa dibaca: {path}")
    h, w = frames[0].shape[:2]
    vp = titik_hilang(garis, h, w, P)
    otomatis = vp is not None
    if not otomatis:                                     # cadangan bila tidak ada marka sama sekali
        vp = (0.5 * w, 0.45 * h)
    y_bawah = batas_bawah_jalan([profil_aspal(f, vp, P) for f in frames], vp[1], h, P)
    kal = roi_dari_titik_hilang(vp, w, h, P, y_bawah)
    kal.update(n_segmen=len(garis), otomatis=otomatis)
    return kal


class PenjejakGaris:
    # Haluskan model garis antar-frame (EMA) dan tahan model lama beberapa frame bila garis hilang sesaat.
    def __init__(self, alpha, hold):
        self.alpha, self.hold = alpha, hold
        self.model = {"kiri": None, "kanan": None}
        self.umur = {"kiri": 999, "kanan": 999}

    def update(self, sisi, baru):
        lama = self.model[sisi]
        if baru is not None:
            if lama is not None and self.umur[sisi] <= self.hold:
                baru = (self.alpha * baru[0] + (1 - self.alpha) * lama[0],
                        self.alpha * baru[1] + (1 - self.alpha) * lama[1])
            self.model[sisi], self.umur[sisi] = baru, 0
            return baru
        self.umur[sisi] += 1
        return lama if (lama is not None and self.umur[sisi] <= self.hold) else None


def titik_garis(model, y_bawah, y_atas, lebar):
    if model is None:
        return None
    return (int(np.clip(model[0] * y_bawah + model[1], -lebar, 2 * lebar)), int(y_bawah),
            int(np.clip(model[0] * y_atas + model[1], -lebar, 2 * lebar)), int(y_atas))


def gambar_hasil(frame, model_kiri, model_kanan, kal, teks=""):
    h, w = frame.shape[:2]
    roi = kal["roi"]
    y_bawah, y_atas = roi[0][1] * h, roi[1][1] * h
    out, lap = frame.copy(), frame.copy()
    gk, gr = titik_garis(model_kiri, y_bawah, y_atas, w), titik_garis(model_kanan, y_bawah, y_atas, w)
    if gk and gr:
        cv2.fillPoly(lap, [np.array([[gk[0], gk[1]], [gk[2], gk[3]], [gr[2], gr[3]], [gr[0], gr[1]]], np.int32)], (0, 180, 80))
    cv2.addWeighted(lap, 0.30, out, 0.70, 0, out)
    tebal = max(4, int(round(w / 200)))
    if gk: cv2.line(out, (gk[0], gk[1]), (gk[2], gk[3]), (255, 200, 0), tebal, cv2.LINE_AA)   # kiri: biru muda (BGR)
    if gr: cv2.line(out, (gr[0], gr[1]), (gr[2], gr[3]), (0, 60, 255), tebal, cv2.LINE_AA)    # kanan: merah
    ket_k = "YA" if gk else "TIDAK"; ket_r = "YA" if gr else "TIDAK"
    cv2.rectangle(out, (8, 8), (560, 74), (20, 20, 20), -1)
    cv2.putText(out, teks, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(out, f"Garis kiri: {ket_k}", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2, cv2.LINE_AA)
    cv2.putText(out, f"Garis kanan: {ket_r}", (230, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 120, 255), 2, cv2.LINE_AA)
    return out


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def proses_video(path, out_path, kal=None, P=PARAM, smoothing=True, catat=(), label="", progress=None):
    """Proses SELURUH frame video -> out_path (.mp4). Bila kal=None, kalibrasi dilakukan otomatis.
    Mengembalikan (ringkasan, prediksi) dengan prediksi[i] = (model_kiri, model_kanan) untuk frame i di `catat`."""
    kal = kalibrasi_video(path, P) if kal is None else kal
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Video tidak bisa dibuka: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, raw = cap.read()
    if not ok:
        raise ValueError("Frame pertama tidak bisa dibaca")
    H, W = praproses(raw, P)[0].shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter gagal dibuka")
    jejak = PenjejakGaris(P["alpha"], P["hold"])
    catat, prediksi = set(catat), {}
    i = n_k = n_r = n_silang = 0
    t0 = time.perf_counter()
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        frame, gray, blur = praproses(raw, P)
        seg = segmentasi(blur, kal, P=P)
        d = deteksi(seg["roi_edges"], gray, kal, P)
        if smoothing:
            mk, mr = jejak.update("kiri", d["model_kiri"]), jejak.update("kanan", d["model_kanan"])
        else:
            mk, mr = d["model_kiri"], d["model_kanan"]
        writer.write(gambar_hasil(frame, mk, mr, kal, f"{label} frame {i}/{n_total}".strip()))
        n_k += mk is not None; n_r += mr is not None; n_silang += bool(d["menyilang"])
        if i in catat:
            prediksi[i] = (mk, mr)
        i += 1
        if progress and (i % 10 == 0 or i == n_total):
            progress(i, n_total)
    cap.release(); writer.release()
    dt = time.perf_counter() - t0
    ringkasan = dict(frame_dibaca=i, ukuran=f"{W}x{H}", fps_video=round(fps, 1), fps_proses=round(i / max(dt, 1e-9), 1),
                     waktu_detik=round(dt, 1), garis_kiri_digambar_pct=round(100 * n_k / max(i, 1), 1),
                     garis_kanan_digambar_pct=round(100 * n_r / max(i, 1), 1),
                     frame_menyilang_pct=round(100 * n_silang / max(i, 1), 1), file=out_path)
    return ringkasan, prediksi
