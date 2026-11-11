"""Deteksi garis marka lajur pada video apa saja, tanpa mengatur parameter.

Pemakaian:
    python deteksi.py video_jalan.mp4                  # hasil -> hasil_video_jalan.mp4
    python deteksi.py video_jalan.mp4 -o keluaran.mp4
    python deteksi.py video_jalan.mp4 --tanpa-smoothing

ROI, baris acuan, dan ambang Canny dihitung otomatis dari videonya sendiri (lihat lane_pipeline.py).
Asumsi rekaman: kamera menghadap lurus ke depan dari kendaraan (dasbor/stang) dan jalan mengisi
bagian bawah frame, sesuai ketentuan pengambilan dataset UTS.
"""
import argparse
import os
import sys

import lane_pipeline as lp


def main():
    ap = argparse.ArgumentParser(description="Deteksi garis marka lajur kiri/kanan pada video (otomatis).")
    ap.add_argument("video", help="path video masukan (.mp4/.mov/.avi)")
    ap.add_argument("-o", "--output", help="path video hasil (.mp4). Bawaan: hasil_<nama>.mp4 di folder yang sama")
    ap.add_argument("--tanpa-smoothing", action="store_true", help="matikan smoothing antar-frame")
    a = ap.parse_args()

    if not os.path.exists(a.video):
        sys.exit(f"Berkas tidak ditemukan: {a.video}")
    nama = os.path.splitext(os.path.basename(a.video))[0]
    out = a.output or os.path.join(os.path.dirname(os.path.abspath(a.video)), f"hasil_{nama}.mp4")

    print("Kalibrasi otomatis (mencari titik hilang)...")
    kal = lp.kalibrasi_video(a.video)
    vx, vy = kal["vp"]
    status = "otomatis" if kal["otomatis"] else "GAGAL, memakai ROI bawaan"
    print(f"  titik hilang = ({vx:.0f}, {vy:.0f}) px [{status}], {kal['n_segmen']} segmen marka dipakai")

    def kemajuan(i, n):
        print(f"\r  memproses frame {i}/{n}", end="", flush=True)

    r, _ = lp.proses_video(a.video, out, kal, smoothing=not a.tanpa_smoothing, progress=kemajuan)
    print()
    print(f"Selesai: {r['frame_dibaca']} frame dalam {r['waktu_detik']} detik ({r['fps_proses']} FPS) -> {out}")
    print(f"  garis kiri digambar {r['garis_kiri_digambar_pct']}% frame, kanan {r['garis_kanan_digambar_pct']}% frame")
    print("  Catatan: persentase ini BUKAN akurasi. Periksa video hasilnya secara visual.")


if __name__ == "__main__":
    main()
