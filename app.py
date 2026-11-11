"""Website Streamlit - Deteksi Marka Jalan pada Video (UTS Visi Komputer, Kelompok 06).

Fitur: unggah video -> video asli dan video hasil deteksi ditampilkan berdampingan -> informasi hasil
(titik hilang, ROI, persentase frame dengan garis kiri/kanan, kecepatan proses). Tidak ada angka yang perlu diatur.

Jalankan lokal:  pip install -r requirements.txt  lalu  streamlit run app.py
Streamlit hanya dipakai sebagai antarmuka; seluruh pengolahan citra ada di lane_pipeline.py (cv2 + numpy).
"""
import os
import shutil
import subprocess
import tempfile

import cv2
import streamlit as st

import lane_pipeline as lp


def cari_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ke_h264(src, dst):
    """Ubah mp4v (tidak bisa diputar di browser) menjadi H.264. True bila berhasil."""
    exe = cari_ffmpeg()
    if exe is None:
        return False
    cmd = [exe, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", dst]
    try:
        subprocess.run(cmd, check=True, timeout=1800)
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


def pratinjau(path, nomor, kal):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, nomor)
    ok, raw = cap.read(); cap.release()
    if not ok:
        return None
    frame, gray, blur = lp.praproses(raw)
    seg = lp.segmentasi(blur, kal)
    d = lp.deteksi(seg["roi_edges"], gray, kal)
    asli = frame.copy()
    cv2.polylines(asli, [seg["pts"]], True, (0, 255, 255), 3)
    cv2.circle(asli, (int(kal["vp"][0]), int(kal["vp"][1])), 10, (255, 0, 255), -1)
    hasil = lp.gambar_hasil(frame, d["model_kiri"], d["model_kanan"], kal, f"frame {nomor}")
    return dict(asli=lp.rgb(frame), roi=lp.rgb(asli), canny=seg["roi_edges"], hasil=lp.rgb(hasil))


st.set_page_config(page_title="Deteksi Garis Marka Lajur - Kelompok 6", layout="wide")
st.title("Deteksi Garis Marka Lajur pada Video")
st.caption("Unggah video jalan yang direkam dari kendaraan (kamera menghadap ke depan). ROI dan ambang dihitung otomatis.")

berkas = st.file_uploader("Unggah video", type=["mp4", "mov", "avi", "mkv", "m4v"])
if berkas is None:
    st.stop()

if st.session_state.get("nama_berkas") != (berkas.name, berkas.size):
    if st.session_state.get("folder"):
        shutil.rmtree(st.session_state["folder"], ignore_errors=True)
    folder = tempfile.mkdtemp(prefix="lane_")
    path_in = os.path.join(folder, "input" + os.path.splitext(berkas.name)[1])
    with open(path_in, "wb") as f:
        f.write(berkas.getbuffer())
    with st.spinner("Kalibrasi otomatis (mencari titik hilang dan batas jalan)..."):
        try:
            kal = lp.kalibrasi_video(path_in)
        except ValueError as e:
            st.error(f"Video tidak bisa dibaca: {e}")
            st.stop()
    cap = cv2.VideoCapture(path_in)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    st.session_state.update(nama_berkas=(berkas.name, berkas.size), path_in=path_in, folder=folder, kal=kal, n=n,
                            meta=dict(fps=fps, w=w0, h=h0), hasil=None)

path_in, kal, n, meta = st.session_state["path_in"], st.session_state["kal"], st.session_state["n"], st.session_state["meta"]
st.write(f"**{berkas.name}** — {meta['w']}x{meta['h']}, {meta['fps']:.1f} FPS, {n} frame (~{n / meta['fps']:.1f} detik)")
if not kal["otomatis"]:
    st.warning("Titik hilang tidak ditemukan (marka tidak terlihat jelas). ROI bawaan dipakai; hasil mungkin kurang baik.")

st.subheader("1. Pratinjau satu frame")
if n > 1:
    nomor = st.slider("Nomor frame", 0, n - 1, n // 2)
    pv = pratinjau(path_in, nomor, kal)
    if pv:
        c1, c2 = st.columns(2)
        c1.image(pv["asli"], caption="Gambar asli")
        c2.image(pv["hasil"], caption="Hasil deteksi (biru = garis kiri, merah = garis kanan)")
        with st.expander("Lihat tahapan antara: ROI otomatis dan tepi Canny"):
            e1, e2 = st.columns(2)
            e1.image(pv["roi"], caption="ROI trapesium otomatis (kuning) + titik hilang (magenta)")
            e2.image(pv["canny"], caption="Tepi Canny di dalam ROI dan masker marka terang")

st.subheader("2. Proses seluruh video")

if st.button("Proses seluruh video", type="primary"):
    out_raw = os.path.join(st.session_state["folder"], "hasil_mp4v.mp4")
    out_h264 = os.path.join(st.session_state["folder"], "hasil_h264.mp4")
    bar = st.progress(0.0, text="Memproses frame...")
    try:
        r, _ = lp.proses_video(path_in, out_raw, kal,
                               progress=lambda i, t: bar.progress(min(i / max(t, 1), 1.0), text=f"Memproses frame {i}/{t}"))
    except (ValueError, RuntimeError) as e:
        bar.empty(); st.error(f"Gagal memproses video: {e}"); st.stop()
    bar.progress(1.0, text="Mengonversi ke H.264 agar bisa diputar di browser...")
    h264 = ke_h264(out_raw, out_h264)
    asli_h264 = os.path.join(st.session_state["folder"], "asli_h264.mp4")   # video HP (HEVC/.mov) sering tidak bisa diputar browser
    asli_ok = ke_h264(path_in, asli_h264)
    bar.empty()
    st.session_state["hasil"] = dict(r=r, path=out_h264 if h264 else out_raw, h264=h264,
                                     asli=asli_h264 if asli_ok else path_in)

hasil = st.session_state.get("hasil")
if hasil:
    r = hasil["r"]
    st.success(f"Selesai: {r['frame_dibaca']} frame diproses dalam {r['waktu_detik']} detik ({r['fps_proses']} FPS).")

    st.subheader("3. Video asli dan hasil deteksi")
    if hasil["h264"]:
        k1, k2 = st.columns(2)
        k1.markdown("**Video asli**")
        k1.video(hasil["asli"])
        k2.markdown("**Video hasil deteksi**")
        k2.video(hasil["path"])
    else:
        st.warning("ffmpeg tidak ditemukan, video hasil memakai codec mp4v yang mungkin tidak bisa diputar di browser. "
                   "Unduh videonya, atau pasang `pip install imageio-ffmpeg`.")

    st.subheader("4. Informasi hasil")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frame diproses", r["frame_dibaca"])
    m2.metric("Garis kiri terdeteksi", f"{r['garis_kiri_digambar_pct']}% frame")
    m3.metric("Garis kanan terdeteksi", f"{r['garis_kanan_digambar_pct']}% frame")
    m4.metric("Kecepatan proses", f"{r['fps_proses']} FPS")
    vx, vy = kal["vp"]
    st.markdown(
        f"- **Titik hilang (otomatis):** ({vx:.0f}, {vy:.0f}) px pada frame {r['ukuran']}\n"
        f"- **ROI trapesium:** tinggi {kal['roi'][1][1]:.2f}–{kal['roi'][0][1]:.2f} dari tinggi frame\n"
        f"- **Frame dengan garis kiri-kanan menyilang (dibuang otomatis):** {r['frame_menyilang_pct']}%\n"
        f"- **Video:** {r['fps_video']} FPS, durasi proses {r['waktu_detik']} detik")
    with open(hasil["path"], "rb") as f:
        st.download_button("Unduh video hasil (.mp4)", f, file_name=f"hasil_{os.path.splitext(berkas.name)[0]}.mp4", mime="video/mp4")
    st.caption("Persentase garis terdeteksi bukan akurasi: video unggahan tidak punya ground truth. Periksa videonya secara visual.")
