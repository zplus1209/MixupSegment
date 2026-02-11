#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def build_class_stats(
    df: pd.DataFrame,
    id_col: str = "finding",
    name_col: str = "finding_name",
    image_col: str = "image_id",
) -> Tuple[List[str], Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Trả về:
      - classes: list tên lớp theo thứ tự new_id (0..C-1)
      - old2new: map id_gốc -> new_id (0..C-1)
      - new2old: map new_id -> id_gốc
      - counts_obj_new: map new_id -> số lượng đối tượng (đếm theo số dòng)
      - counts_img_new: map new_id -> số lượng ảnh duy nhất có chứa lớp đó
    """
    if id_col not in df.columns:
        raise ValueError(f"CSV thiếu cột '{id_col}'")
    if image_col and image_col not in df.columns:
        raise ValueError(f"CSV thiếu cột '{image_col}'")

    # Lấy id gốc (có thể là số/chuỗi). Nếu ép int lỗi thì factorize trực tiếp.
    try:
        raw_ids = df[id_col].astype(int).to_numpy()
        factorized = False
    except Exception:
        raw_ids, uniques = pd.factorize(df[id_col])
        # 'raw_ids' lúc này là chỉ số (0..K-1) theo thứ tự xuất hiện; ta vẫn cần map -> new_id theo sort tăng.
        factorized = True

    # Tập id gốc duy nhất, sắp tăng
    uniq_old = np.unique(raw_ids)
    uniq_old_sorted = np.sort(uniq_old)

    # Ánh xạ id_gốc -> new_id 0..C-1
    old2new = {int(old): int(i) for i, old in enumerate(uniq_old_sorted)}
    new2old = {int(i): int(old) for i, old in enumerate(uniq_old_sorted)}

    # Tên lớp
    classes: List[str] = []
    if name_col in df.columns:
        for old in uniq_old_sorted:
            if factorized:
                idx = np.where(raw_ids == old)[0]
                nm = str(df.iloc[idx[0]][name_col]) if len(idx) > 0 else f"class_{old}"
            else:
                tmp = df.loc[df[id_col].astype(int) == int(old), name_col]
                nm = str(tmp.iloc[0]) if len(tmp) > 0 else f"class_{old}"
            classes.append(nm)
    else:
        classes = [f"class_{old}" for old in uniq_old_sorted]

    # Đếm số lượng theo new_id
    counts_obj_new: Dict[int, int] = {}  # số đối tượng (số dòng)
    counts_img_new: Dict[int, int] = {}  # số ảnh duy nhất

    # Tiền xử lý mảng image_id cho nhanh
    image_ids = df[image_col].to_numpy()

    for new_id in range(len(classes)):
        old = new2old[new_id]

        # Đếm object: số dòng có id_gốc == old
        mask = (raw_ids == old)
        obj_count = int(mask.sum())
        counts_obj_new[new_id] = obj_count

        # Đếm image duy nhất: số lượng image_id distinct trong các dòng thuộc lớp
        img_count = int(len(np.unique(image_ids[mask])))
        counts_img_new[new_id] = img_count

    return classes, old2new, new2old, counts_obj_new, counts_img_new


def save_outputs(
    out_dir: str,
    classes: List[str],
    old2new: Dict[int, int],
    new2old: Dict[int, int],
    counts_obj_new: Dict[int, int],
    counts_img_new: Dict[int, int],
    id_col: str,
    name_col: str,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) classes.json
    classes_path = os.path.join(out_dir, "classes.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, ensure_ascii=False, indent=2)

    # 2) mapping.json
    mapping_path = os.path.join(out_dir, "mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "id_col": id_col,
                "name_col": name_col,
                "old2new": old2new,
                "new2old": new2old,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 3) class_counts.csv (bảng new_id, class_name, old_id, object_count, image_count)
    rows = []
    for new_id in range(len(classes)):
        rows.append(
            {
                "new_id": new_id,
                "class_name": classes[new_id],
                "old_id": new2old[new_id],
                "object_count": counts_obj_new[new_id],
                "image_count": counts_img_new[new_id],
            }
        )
    counts_df = pd.DataFrame(rows).sort_values("new_id")
    counts_csv_path = os.path.join(out_dir, "class_counts.csv")
    counts_df.to_csv(counts_csv_path, index=False)

    # 4) Tổng quan (tuỳ chọn): tổng số đối tượng/ảnh
    totals_path = os.path.join(out_dir, "totals.json")
    with open(totals_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_classes": len(classes),
                "total_objects": int(sum(counts_obj_new.values())),
                "total_images": int(sum(counts_img_new.values())),  # lưu ý: cộng per-class (không phải distinct toàn cục)
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"✓ Saved: {classes_path}")
    print(f"✓ Saved: {mapping_path}")
    print(f"✓ Saved: {counts_csv_path}")
    print(f"✓ Saved: {totals_path}")


def print_python_snippet(
    classes: List[str],
    old2new: Dict[int, int],
    new2old: Dict[int, int],
):
    print("\n# ---------- PASTE-READY PYTHON SNIPPET ----------")
    print("classes = [")
    for nm in classes:
        print(f"    {repr(nm)},")
    print("]\n")
    print(f"old2new = {old2new}")
    print(f"new2old = {new2old}")
    print("# Example usage:")
    print("# memory_loader.dataset.classes = classes")
    print("# memory_loader.dataset.class_to_idx = {name: i for i, name in enumerate(classes)}")
    print("# memory_loader.dataset.idx_to_class = {i: name for i, name in enumerate(classes)}")
    print("# # If you want to reindex labels (0..C-1) on an existing dataset with old ids:")
    print("# ds = memory_loader.dataset")
    print("# ds.labels = [old2new[int(y)] for y in ds.labels]  # or ds.targets")


def main():
    ap = argparse.ArgumentParser(description="Summarize class mapping from CSV.")
    ap.add_argument("--csv", required=True, help="Đường dẫn CSV (có cột finding/finding_name)")
    ap.add_argument("--id-col", default="finding", help="Tên cột id lớp (mặc định: finding)")
    ap.add_argument("--name-col", default="finding_name", help="Tên cột tên lớp (mặc định: finding_name, có thể không có)")
    ap.add_argument("--image-col", default="image_id", help="Tên cột id ảnh (mặc định: image_id)")
    ap.add_argument("--out-dir", default="class_summary", help="Thư mục xuất kết quả (json/csv)")
    ap.add_argument("--no-save", action="store_true", help="Không lưu file, chỉ in ra màn hình")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    classes, old2new, new2old, counts_obj_new, counts_img_new = build_class_stats(
        df, id_col=args.id_col, name_col=args.name_col, image_col=args.image_col
    )

    # In tóm tắt
    print(f"Num classes: {len(classes)}")
    for i, nm in enumerate(classes):
        print(
            f"[{i}] {nm}  (old_id={new2old[i]}, "
            f"object_count={counts_obj_new[i]}, image_count={counts_img_new[i]})"
        )

    if not args.no_save:
        save_outputs(
            args.out_dir,
            classes,
            old2new,
            new2old,
            counts_obj_new,
            counts_img_new,
            args.id_col,
            args.name_col,
        )

    # Đưa snippet để copy/paste vào code
    print_python_snippet(classes, old2new, new2old)


if __name__ == "__main__":
    main()
