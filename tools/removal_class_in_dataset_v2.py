#!/usr/bin/env python3
import argparse, os, json, glob
import pandas as pd

DEFAULT_WHITELIST = [
    # Giống ảnh: thứ tự chính là new_id
    "bbps-0-1",
    "bbps-2-3",
    "cecum",
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis-a",
    "esophagitis-b-d",
    "impacted-stool",
    "polyps",
    "pylorus",
    "retroflex-rectum",
    "retroflex-stomach",
    "ulcerative-colitis-grade-1",
    "ulcerative-colitis-grade-2",
    "ulcerative-colitis-grade-3",
    "z-line",
]

normal_classes = [
    "cecum",
    "ileum",
    "pylorus",
    "retroflex-rectum",
    "retroflex-stomach",
    "z-line",
]

abnormal_classes = [
    "barretts",
    "barretts-short-segment",
    "esophagitis-a",
    "esophagitis-b-d",
    "hemorrhoids",
    "polyps",
    "ulcerative-colitis-grade-0-1",
    "ulcerative-colitis-grade-1",
    "ulcerative-colitis-grade-1-2",
    "ulcerative-colitis-grade-2",
    "ulcerative-colitis-grade-2-3",
    "ulcerative-colitis-grade-3",
]

BINARY_NAME2ID = {"normal": 0, "abnormal": 1}

def read_whitelist(path: str):
    """
    Hỗ trợ:
      - .txt: mỗi dòng 1 tên lớp
      - .csv: ưu tiên cột 'class_name' hoặc cột đầu tiên
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    elif ext == ".csv":
        df = pd.read_csv(path)
        if "class_name" in df.columns:
            return df["class_name"].astype(str).tolist()
        # cột đầu tiên
        return df.iloc[:, 0].astype(str).tolist()
    else:
        raise ValueError(f"Unsupported whitelist file: {path}")

def collect_files(src_dir: str):
    pats = [os.path.join(src_dir, "train_endo*.csv"),
            os.path.join(src_dir, "val_endo*.csv")]
    files = []
    for p in pats:
        files += sorted(glob.glob(p))
    return files

def load_concat(files):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # chuẩn cột
        need_cols = {"image_id","finding_name","finding","normal_abnormal","binary_label"}
        miss = need_cols - set(df.columns)
        if miss:
            raise ValueError(f"{f} thiếu cột: {miss}")
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True)

def choose_classes(df_all, args):
    if args.whitelist:
        classes = read_whitelist(args.whitelist)
    elif args.use_default_whitelist:
        classes = list(DEFAULT_WHITELIST)
    elif args.min_images is not None:
        ct = df_all["finding_name"].value_counts().sort_values(ascending=False)
        classes = ct[ct >= args.min_images].index.tolist()
        if args.order == "name":
            classes = sorted(classes)
        elif args.order == "count":
            # đã theo count desc
            pass
        else:
            # giữ nguyên
            pass
    else:
        # fallback: dùng default whitelist
        classes = list(DEFAULT_WHITELIST)

    # Giữ nguyên thứ tự whitelist (nếu có), đồng thời chỉ lấy các class thực sự có trong dữ liệu
    present = set(df_all["finding_name"].unique())
    selected = [c for c in classes if c in present]
    return selected

def build_mapping(selected_classes):
    old_order = selected_classes[:]  # tên lớp ~ old label space by name
    # new_id = index trong danh sách
    new2name = {i: name for i, name in enumerate(old_order)}
    name2new = {name: i for i, name in new2name.items()}
    return name2new, new2name

def reindex_and_save_one(df, name2new, dst_path):
    keep = df["finding_name"].isin(name2new.keys())
    out = df.loc[keep].copy()
    out["finding"] = out["finding_name"].map(name2new).astype(int)
    # Giữ các cột còn lại nguyên vẹn
    out.to_csv(dst_path, index=False)

# =======================
# Binary conversion (2 cột: finding_name, finding)
# =======================
def _map_class_to_binary_label(cls: str):
    if cls in normal_classes:
        return "normal"
    if cls in abnormal_classes:
        return "abnormal"
    return None  # không xác định => loại bỏ

def convert_to_binary_one(df: pd.DataFrame, dst_path: str):
    """
    - Lọc trước các hàng không thuộc DEFAULT_WHITELIST
    - Chỉ giữ 3 cột: image_id, finding_name ('normal' hoặc 'abnormal') và finding (0/1)
    - Bỏ các lớp không map được (vd bbps-*, dyed-*)
    """
    # 1) Chỉ giữ lớp thuộc whitelist
    df_w = df[df["finding_name"].isin(DEFAULT_WHITELIST)].copy()

    # 2) Map lớp gốc => 'normal' / 'abnormal'
    bin_name = df_w["finding_name"].map(_map_class_to_binary_label)

    # 3) Loại bỏ hàng không map được
    mask = bin_name.isin(BINARY_NAME2ID.keys())
    df_w = df_w.loc[mask].copy()

    # 4) Gán cột nhị phân theo yêu cầu
    df_w["finding_name"] = bin_name.loc[mask]                 # 'normal' / 'abnormal'
    df_w["finding"] = df_w["finding_name"].map(BINARY_NAME2ID).astype(int)  # 0/1

    # 5) Chỉ xuất 3 cột để tiện tra cứu theo image_id
    out = df_w[["image_id", "finding_name", "finding"]].copy()
    out.to_csv(dst_path, index=False)

def main():
    ap = argparse.ArgumentParser(description="Filter weak classes and remap labels.")
    ap.add_argument("--src-dir", default="dataset/full_class",
                    help="Thư mục chứa các CSV train_endo*.csv / val_endo*.csv")
    ap.add_argument("--dst-dir", default="dataset/removal_weak_class",
                    help="Thư mục ghi kết quả CSV đã lọc")
    ap.add_argument("--binary-cls", default="dataset/binary_class",
                    help="Thư mục ghi kết quả CSV lớp nhị phân (2 cột).")
    # chế độ chọn lớp
    ap.add_argument("--use-default-whitelist", action="store_true",
                    help="Dùng danh sách 16 lớp như ảnh (mặc định nếu không chỉ định gì).")
    ap.add_argument("--whitelist", type=str, default=None,
                    help="Đường dẫn file .txt/.csv chứa danh sách lớp (mỗi dòng 1 tên).")
    ap.add_argument("--min-images", type=int, default=None,
                    help="Chọn lớp có số ảnh ≥ ngưỡng này (thay cho whitelist).")
    ap.add_argument("--order", type=str, default="whitelist", choices=["whitelist","name","count"],
                    help="Thứ tự khi dùng --min-images: theo tên, theo số ảnh, hay giữ nguyên (whitelist).")
    args = ap.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    os.makedirs(args.binary_cls, exist_ok=True)

    files = collect_files(args.src_dir)
    if not files:
        raise SystemExit(f"Không tìm thấy CSV trong {args.src_dir}")

    df_all = load_concat(files)
    selected = choose_classes(df_all, args)
    if not selected:
        raise SystemExit("Không có lớp nào được chọn sau khi lọc!")

    name2new, new2name = build_mapping(selected)

    # Tổng hợp thống kê theo new_id
    sub = df_all[df_all["finding_name"].isin(selected)].copy()
    counts = (
        sub.groupby("finding_name")["image_id"]
        .nunique()
        .reset_index(name="image_count")
    )
    counts["new_id"] = counts["finding_name"].map(name2new)
    counts = counts.sort_values("new_id")
    counts["class_name"] = counts["finding_name"]
    counts = counts[["new_id","class_name","image_count"]]

    # Lưu mapping & classes
    with open(os.path.join(args.dst_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": [new2name[i] for i in range(len(new2name))]}, f,
                  ensure_ascii=False, indent=2)
    with open(os.path.join(args.dst_dir, "mapping_name2new.json"), "w", encoding="utf-8") as f:
        json.dump(name2new, f, ensure_ascii=False, indent=2)
    counts.to_csv(os.path.join(args.dst_dir, "class_counts.csv"), index=False)

    # Ghi lại từng file CSV với nhãn mới (multiclass, giữ nguyên cấu trúc cũ)
    for f in files:
        df = pd.read_csv(f)
        dst_path = os.path.join(args.dst_dir, os.path.basename(f))
        reindex_and_save_one(df, name2new, dst_path)

    # ======= Tạo bộ binary (2 cột) từ dữ liệu gốc theo yêu cầu =======
    # - Lọc theo DEFAULT_WHITELIST trước
    # - Map thành normal/abnormal
    # - Chỉ xuất 2 cột finding_name, finding
    for f in files:
        df = pd.read_csv(f)
        dst_bin = os.path.join(args.binary_cls, os.path.basename(f))
        convert_to_binary_one(df, dst_bin)

    # Lưu thêm mapping nhị phân để tham chiếu
    with open(os.path.join(args.binary_cls, "binary_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"name2id": BINARY_NAME2ID, "id2name": {v:k for k,v in BINARY_NAME2ID.items()}},
                  f, ensure_ascii=False, indent=2)

    print("✓ Done.")
    print(f"- Selected {len(selected)} classes:")
    for i, name in enumerate(selected):
        print(f"  [{i}] {name}")
    print(f"- Saved filtered CSVs to: {args.dst_dir}")
    print(f"- Saved summary: classes.json, mapping_name2new.json, class_counts.csv")
    print(f"- Saved binary CSVs (2 cols) to: {args.binary_cls} (only normal/abnormal)")
    print("  Note: các lớp trong DEFAULT_WHITELIST nhưng không thuộc normal/abnormal sẽ bị loại ở binary.")
    
if __name__ == "__main__":
    main()
