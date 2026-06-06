from __future__ import annotations
import csv
from datetime import datetime
import io
import os
import re
import random
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

TARGET_SOURCES = [
    "Gloves&Scarves.csv",
    "Jewelry.csv",
    "Jumpsuits&Romppers Non-Curve.csv",
    "Jumpsuits&Romppers.csv",
    "Sweater.csv",
    "Tops.csv",

    "Part 4 Data",
    "phase2 dataP3",
    "part5 data",
    "TopsLV.csv",
    "Dresses-4-Done.csv",
    "Tanks&Vests-Done.csv",
    "Hats 2-Done.csv",
    "Fashion-Glasses 2-Done.csv",
]

BODY_SHAPES = ["كمثري", "تفاحة", "ساعة رملية", "مستطيل", "مثلث مقلوب"]
OCCASIONS = ["أي مناسبة", "عمل", "يومي", "حفلة", "سفر"]
STYLES = ["أي نمط", "يومي", "كلاسيكي", "أنيق", "عصري", "بسيط"]
LOOK_TYPES = ["بلوزة + بنطالون", "بلوزة + تنورة", "جمبسوت", "فستان", "طقم متناسق"]

def normalize_user_occasion(value):
    mapping = {
        "أي مناسبة": "أي مناسبة",
        "العمل": "work",
        "عمل": "work",
        "كاجوال": "casual",
        "يومي": "casual",
        "حفلة": "party",
        "سفر": "travel",
    }
    return mapping.get(value, value)


def normalize_user_style(value):
    mapping = {
        "أي ستايل": "أي ستايل",
        "أي نمط": "أي نمط",
        "كاجوال": "casual",
        "يومي": "casual",
        "كلاسيك": "classic",
        "كلاسيكي": "classic",
        "شيك": "chic",
        "أنيق": "chic",
        "مودرن": "modern",
        "عصري": "modern",
        "مينمل": "minimal",
        "بسيط": "minimal",
    }
    return mapping.get(value, value)

COLOR_HARMONY = {
    "Black": ["White", "Beige", "Grey", "Pink", "Blue"],
    "White": ["Black", "Blue", "Beige", "Grey", "Pink"],
    "Blue": ["White", "Beige", "Grey", "Black"],
    "Navy": ["White", "Beige", "Grey"],
    "Beige": ["Black", "White", "Blue", "Green", "Pink"],
    "Grey": ["Black", "White", "Blue", "Pink"],
    "Pink": ["White", "Beige", "Grey"],
    "Green": ["White", "Beige", "Black"],
    "Red": ["Black", "White", "Beige"],
    "Burgundy": ["Beige", "White", "Black"],
    "Orange": ["White", "Beige", "Brown"],
    "Yellow": ["White", "Grey", "Beige"],
    "Gold": ["Black", "White", "Beige"],
    "Silver": ["Black", "White", "Grey"],
    "Multicolor": ["Black", "White", "Beige", "Grey"],
    "Brown": ["Beige", "White", "Black", "Gold"],
}

ACCESSORY_COLOR_HARMONY = {
    "Black": ["Black", "Beige", "White", "Grey", "Gold", "Silver"],
    "White": ["White", "Beige", "Black", "Silver", "Grey"],
    "Beige": ["Beige", "White", "Brown", "Gold", "Black"],
    "Brown": ["Brown", "Beige", "Gold", "Black"],
    "Blue": ["Blue", "White", "Silver", "Beige", "Black"],
    "Navy": ["Navy", "Beige", "White", "Silver", "Black"],
    "Grey": ["Grey", "White", "Black", "Silver", "Beige"],
    "Pink": ["Pink", "Beige", "White", "Silver", "Gold"],
    "Red": ["Black", "Beige", "Gold", "White"],
    "Burgundy": ["Black", "Beige", "Gold", "White"],
    "Green": ["Beige", "White", "Gold", "Black"],
    "Yellow": ["Beige", "White", "Gold", "Brown"],
    "Orange": ["Beige", "Brown", "Gold", "White"],
    "Gold": ["Gold", "Beige", "Black", "White"],
    "Silver": ["Silver", "Grey", "White", "Black"],
    "Multicolor": ["Black", "Beige", "White", "Grey"],
}


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def lower_text(x) -> str:
    return clean_text(x).lower()


def read_csv_robust(file_obj_or_path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "cp1256"]
    seps = [None, ",", ";"]
    last_error = None

    if hasattr(file_obj_or_path, "read"):
        raw = file_obj_or_path.read()
    else:
        with open(file_obj_or_path, "rb") as f:
            raw = f.read()

    for enc in encodings:
        for sep in seps:
            try:
                bio = io.BytesIO(raw)
                if sep is None:
                    df = pd.read_csv(bio, encoding=enc, sep=None, engine="python")
                else:
                    df = pd.read_csv(bio, encoding=enc, sep=sep)
                if len(df.columns) > 1:
                    return df
            except Exception as e:
                last_error = e

    raise last_error if last_error else ValueError("تعذر قراءة الملف")


def read_excel_robust(path_or_bytes) -> pd.DataFrame:
    if hasattr(path_or_bytes, "read"):
        return pd.read_excel(io.BytesIO(path_or_bytes.read()))
    return pd.read_excel(path_or_bytes)


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = [str(c).strip().lower() for c in df.columns]
    for cand in candidates:
        for i, c in enumerate(cols):
            if cand in c:
                return df.columns[i]
    return None


def normalize_color(value: str) -> str:
    v = lower_text(value)
    if not v or v == "nan":
        return ""

    if "navy" in v:
        return "Navy"
    if "blue" in v:
        return "Blue"
    if any(x in v for x in ["beige", "off-white", "off white", "cream", "creamy", "creamy white", "ivory"]):
        return "Beige"
    if "green olive" in v or "olive" in v:
        return "Green"
    if "green" in v:
        return "Green"
    if "multi" in v:
        return "Multicolor"
    if "red" in v:
        return "Red"
    if any(x in v for x in ["burgundy", "wine", "maroon"]):
        return "Burgundy"
    if "orange" in v:
        return "Orange"
    if "yellow" in v:
        return "Yellow"
    if "white" in v:
        return "White"
    if "pink" in v:
        return "Pink"
    if "gold" in v:
        return "Gold"
    if "silver" in v:
        return "Silver"
    if "black" in v:
        return "Black"
    if "grey" in v or "gray" in v:
        return "Grey"
    if "brown" in v:
        return "Brown"
    return ""


def color_label_ar(color: str) -> str:
    labels = {
        "أي لون": "أي لون",
        "Black": "أسود",
        "White": "أبيض",
        "Grey": "رمادي",
        "Blue": "أزرق",
        "Navy": "كحلي",
        "Beige": "بيج",
        "Green": "أخضر",
        "Multicolor": "متعدد الألوان",
        "Red": "أحمر",
        "Burgundy": "عنابي",
        "Orange": "برتقالي",
        "Yellow": "أصفر",
        "Pink": "وردي",
        "Gold": "ذهبي",
        "Silver": "فضي",
        "Brown": "بني",
    }
    return labels.get(color, color)


def normalize_occasion(value: str) -> str:
    v = lower_text(value)
    if not v:
        return ""

    if any(x in v for x in ["casual", "daily", "daywear", "everyday"]):
        return "يومي"
    if any(x in v for x in ["work", "office", "formal work", "college", "campus", "university", "school"]):
        return "عمل"
    if any(x in v for x in ["party", "evening", "night out", "event"]):
        return "حفلة"
    if any(x in v for x in ["travel", "airport", "vacation"]):
        return "سفر"
    return ""


def infer_occasion(title: str, source_file: str) -> str:
    text = f"{lower_text(title)} {lower_text(source_file)}"

    if any(x in text for x in ["party", "sequin", "satin", "velvet", "lace", "dressy", "evening"]):
        return "حفلة"
    if any(x in text for x in ["office", "tailored", "formal", "blazer", "work", "college", "campus", "school", "university"]):
        return "عمل"
    if any(x in text for x in ["travel", "airport", "vacation"]):
        return "سفر"
    return "يومي"


def infer_style(title: str, source_file: str) -> str:
    text = f"{lower_text(title)} {lower_text(source_file)}"

    if any(x in text for x in ["classic", "tailored", "elegant", "formal", "structured"]):
        return "كلاسيكي"
    if any(x in text for x in ["minimal", "basic", "clean", "simple"]):
        return "بسيط"
    if any(x in text for x in ["chic", "feminine", "satin", "lace", "dressy"]):
        return "أنيق"
    if any(x in text for x in ["modern", "trendy", "fashion", "statement"]):
        return "عصري"
    return "يومي"


def build_frame_from_df(df: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
    title_col = find_col(df, ["title", "product name", "name"])
    color_col = find_col(df, ["color", "colour"])
    image_col = find_col(df, ["image url", "image_url", "image"])
    url_col = find_col(df, ["product url", "product_url", "url"])
    occasion_col = find_col(df, ["occasion"])

    if title_col is None:
        return None

    return pd.DataFrame({
        "title": df[title_col].map(clean_text),
        "color": df[color_col].map(clean_text) if color_col else "",
        "image_url": df[image_col].map(clean_text) if image_col else "",
        "product_url": df[url_col].map(clean_text) if url_col else "",
        "occasion_raw": df[occasion_col].map(clean_text) if occasion_col else "",
        "source_file": source_name,
    })


def iter_supported_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".csv", ".xlsx", ".xls"]:
            yield p


def load_all_data() -> pd.DataFrame:
    frames = []

    for source_name in TARGET_SOURCES:
        path = BASE_DIR / source_name
        if not path.exists():
            continue

        try:
            if path.is_dir():
                for file_path in iter_supported_files(path):
                    try:
                        if file_path.suffix.lower() == ".csv":
                            df = read_csv_robust(file_path)
                        else:
                            df = read_excel_robust(file_path)

                        out = build_frame_from_df(df, file_path.name)
                        if out is not None:
                            frames.append(out)
                    except Exception:
                        continue

            elif path.is_file():
                if path.suffix.lower() == ".zip":
                    with zipfile.ZipFile(path, "r") as zf:
                        for member in zf.namelist():
                            if member.endswith("/"):
                                continue
                            member_lower = member.lower()
                            try:
                                if member_lower.endswith(".csv"):
                                    with zf.open(member) as f:
                                        df = read_csv_robust(f)
                                elif member_lower.endswith(".xlsx") or member_lower.endswith(".xls"):
                                    with zf.open(member) as f:
                                        df = read_excel_robust(f)
                                else:
                                    continue

                                out = build_frame_from_df(df, os.path.basename(member))
                                if out is not None:
                                    frames.append(out)
                            except Exception:
                                continue

                elif path.suffix.lower() == ".csv":
                    df = read_csv_robust(path)
                    out = build_frame_from_df(df, path.name)
                    if out is not None:
                        frames.append(out)

                elif path.suffix.lower() in [".xlsx", ".xls"]:
                    df = read_excel_robust(path)
                    out = build_frame_from_df(df, path.name)
                    if out is not None:
                        frames.append(out)

        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=[
            "title", "color", "image_url", "product_url",
            "occasion", "source_file", "style"
        ])

    data = pd.concat(frames, ignore_index=True).drop_duplicates()
    data["color"] = data["color"].map(normalize_color)
    data["occasion"] = data["occasion_raw"].map(normalize_occasion)
    data["occasion"] = data.apply(
        lambda r: r["occasion"] if r["occasion"] else infer_occasion(r["title"], r["source_file"]),
        axis=1,
    )
    data["style"] = data.apply(lambda r: infer_style(r["title"], r["source_file"]), axis=1)
    data["is_curve_plus"] = data["title"].apply(
        lambda x: "curve & plus" in lower_text(x) or "curve plus" in lower_text(x)
    )
    return data

def contains_any_keyword(text: str, keywords: list[str]) -> bool:
    text = lower_text(text)
    for kw in keywords:
        kw = lower_text(kw)
        if kw and kw in text:
            return True
    return False


def contains_any_whole_word(text: str, words: list[str]) -> bool:
    text = lower_text(text)
    for word in words:
        word = re.escape(lower_text(word))
        if re.search(rf"(?<![a-z]){word}(?![a-z])", text):
            return True
    return False

def classify_piece(source_file: str, title: str) -> str:
    src = lower_text(source_file)
    t = lower_text(title)
    combined = f"{src} {t}"

    clothing_words = [
        "dress", "dresses", "top", "tops", "pant", "pants", "trouser", "trousers",
        "skirt", "skirts", "jumpsuit", "romper", "sweater", "blouse", "shirt",
        "coat", "jacket", "hoodie", "jeans", "cardigan", "puffer", "vest",
        "tank", "bodysuit", "tee", "t-shirt", "leggings", "joggers", "shorts"
    ]

    
    if contains_any_keyword(t, ["glasses", "sunglasses", "eyewear"]):
        return "glasses"
    if "fashion-glasses" in src and contains_any_keyword(t, ["glasses", "sunglasses", "frame"]):
        return "glasses"

    
    if not contains_any_whole_word(t, clothing_words):
        if contains_any_whole_word(t, [
            "necklace", "earring", "earrings", "bracelet",
            "ring", "rings", "chain", "pendant", "choker"
        ]):
            return "jewelry"

   
    if not contains_any_whole_word(t, clothing_words):
        if contains_any_whole_word(t, ["scarf", "scarves", "glove", "gloves", "shawl"]):
            return "scarf"

    
    if contains_any_whole_word(t, ["hat", "hats", "cap", "caps"]):
        return "exclude"

    
    if (
        contains_any_keyword(src, ["bags", "bag", "handbag", "crossbody", "tote", "clutch", "satchel"])
        or contains_any_whole_word(t, ["bag", "bags", "handbag", "crossbody", "tote", "clutch", "satchel", "purse"])
    ):
        return "bag"

    
    if (
        contains_any_keyword(src, [
            "shoes", "shoe", "heels", "heel", "sandals", "sandal",
            "boots", "boot", "sneakers", "sneaker", "flats", "flat",
            "loafers", "loafer"
        ])
        or contains_any_whole_word(t, [
            "shoe", "shoes", "heel", "heels", "sandal", "sandals",
            "boot", "boots", "sneaker", "sneakers", "flat", "flats",
            "loafer", "loafers", "mule", "mules", "pump", "pumps"
        ])
    ):
        return "shoes"

    
    if (
        contains_any_keyword(src, ["dresses", "dress"])
        or contains_any_whole_word(t, ["dress", "dresses", "gown"])
    ):
        return "dress"


    if (
        contains_any_keyword(src, ["jumpsuits&rompers", "jumpsuit", "romper"])
        or contains_any_whole_word(t, ["jumpsuit", "romper", "playsuit"])
    ):
        return "jumpsuit"


    if (
        contains_any_keyword(src, [
            "matching-sets", "matching set", "matching sets",
            "co-ord", "coord", "co ord", "two piece", "two-piece",
            "outfit set", "matching outfit", "co-ordinate", "co ordinates"
        ])
        or contains_any_whole_word(t, [
            "set", "sets", "two piece", "two-piece", "matching set",
            "matching sets", "co-ord", "coord", "co ord", "co-ordinate",
            "co ordinates"
        ])
    ):
        return "set"

    
    if (
        contains_any_keyword(src, [
            "topslv", "tops", "sweater", "cardigan",
            "outerwear&jackets", "tanks&vests", "tank", "vest"
        ])
        or contains_any_whole_word(t, [
            "top", "tops", "blouse", "shirt", "sweater",
            "cardigan", "jacket", "coat", "tank", "vest", "shrug", "puffer", "hoodie"
        ])
    ):
        return "top"

    
    if (
        contains_any_keyword(src, ["maxi-skirt", "midi-skirt", "mini-skirt", "skirt"])
        or contains_any_whole_word(t, ["skirt", "skirts", "skort", "skorts"])
    ):
        return "skirt"


    if (
        contains_any_keyword(src, ["jeans", "pants", "trouser", "trousers", "bottoms", "legging", "joggers"])
        or contains_any_whole_word(t, ["jean", "jeans", "pant", "pants", "trouser", "trousers", "legging", "leggings", "jogger", "joggers"])
    ):
        return "pants"

    return "other"

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["piece_type"] = data.apply(lambda r: classify_piece(r["source_file"], r["title"]), axis=1)
    data = data[data["piece_type"] != "exclude"].copy()
    data = data[data["piece_type"].isin([
        "top", "pants", "skirt", "jumpsuit", "dress", "set",
        "bag", "shoes", "glasses", "jewelry", "scarf"
    ])].copy()
    data = data[(data["title"] != "") & (data["image_url"] != "")].copy()
    data = data.drop_duplicates(subset=["title", "image_url", "product_url"])
    return data.reset_index(drop=True)


DATA = prepare_data(load_all_data())


def get_available_colors(piece_type: str) -> list[str]:
    colors = (
        DATA.loc[DATA["piece_type"] == piece_type, "color"]
        .dropna()
        .map(clean_text)
        .tolist()
    )
    colors = sorted({c for c in colors if c})
    return ["أي لون"] + colors

def enrich_item_features(row) -> str:
    piece_type = clean_text(row.get("piece_type", ""))
    title = lower_text(row.get("title", ""))
    extra = []

    if piece_type == "shoes":
        if any(x in title for x in ["heel", "heels", "stiletto", "pump", "pumps", "strappy", "pointed", "slingback", "mule"]):
            extra.extend(["elegant", "formal", "chic", "party", "feminine"])

        if any(x in title for x in ["chunky", "combat", "platform", "lug", "boot", "boots", "sneaker", "sneakers"]):
            extra.extend(["casual", "heavy", "street", "edgy"])

        if any(x in title for x in ["flat", "flats", "loafer", "loafers"]):
            extra.extend(["classic", "daily", "smart", "comfortable"])

    elif piece_type == "bag":
        if any(x in title for x in ["clutch", "mini bag", "shoulder bag", "evening"]):
            extra.extend(["elegant", "party", "formal", "chic"])

        if any(x in title for x in ["tote", "large", "capacity", "shopper"]):
            extra.extend(["daily", "practical", "casual"])

        if any(x in title for x in ["crossbody"]):
            extra.extend(["modern", "casual", "smart"])

    elif piece_type in ["dress", "set", "top", "pants", "skirt", "jumpsuit"]:
        if any(x in title for x in ["satin", "sequin", "lace", "corset", "ruched"]):
            extra.extend(["party", "elegant", "chic", "feminine"])

        if any(x in title for x in ["tailored", "structured", "clean"]):
            extra.extend(["classic", "formal", "smart"])

        if any(x in title for x in ["oversized", "relaxed", "cargo", "baggy"]):
            extra.extend(["casual", "modern", "street"])

    return " ".join(extra)

def build_features(row):
    parts = [
        clean_text(row.get("title", "")),
        clean_text(row.get("color", "")),
        clean_text(row.get("occasion", "")),
        clean_text(row.get("style", "")),
        clean_text(row.get("piece_type", "")),
        clean_text(row.get("source_file", "")),
        enrich_item_features(row),
    ]
    return " ".join([p for p in parts if p])


def get_harmony_colors(color: str) -> list[str]:
    return COLOR_HARMONY.get(color, [])


def get_accessory_priority_colors(base_colors: list[str]) -> list[str]:
    cleaned_base = [clean_text(c) for c in base_colors if clean_text(c)]
    if not cleaned_base:
        return ["Black", "Beige", "White", "Grey", "Gold", "Silver"]

    primary = cleaned_base[0]
    priority = [primary]
    priority.extend(ACCESSORY_COLOR_HARMONY.get(primary, []))

    for c in cleaned_base[1:]:
        if c not in priority:
            priority.append(c)
        for extra in ACCESSORY_COLOR_HARMONY.get(c, []):
            if extra not in priority:
                priority.append(extra)

    cleaned = []
    for c in priority:
        if c and c not in cleaned:
            cleaned.append(c)
    return cleaned


def body_shape_score(item_title: str, piece_type: str, body_shape: str) -> int:
    title = lower_text(item_title)
    score = 0

    if body_shape == "كمثري":
        if piece_type == "top":
            score += 3 if any(x in title for x in ["puff", "ruffle", "shoulder", "printed", "volume"]) else 1
        if piece_type in ["pants", "skirt"]:
            score += 3 if any(x in title for x in ["wide", "straight", "a-line", "aline", "flowy"]) else 1
    elif body_shape == "تفاحة":
        if piece_type == "top":
            score += 3 if any(x in title for x in ["v-neck", "wrap", "longline", "button"]) else 1
        if piece_type in ["pants", "skirt"]:
            score += 2 if any(x in title for x in ["high waist", "straight", "wide"]) else 1
    elif body_shape == "ساعة رملية":
        score += 3 if any(x in title for x in ["fitted", "tailored", "wrap", "belted", "high waist"]) else 1
    elif body_shape == "مستطيل":
        score += 3 if any(x in title for x in ["ruffle", "pleated", "layered", "wide", "flare", "belted"]) else 1
    elif body_shape == "مثلث مقلوب":
        if piece_type in ["pants", "skirt"]:
            score += 3 if any(x in title for x in ["wide", "flare", "pleated", "cargo", "printed"]) else 1
        if piece_type == "top":
            score += 2 if any(x in title for x in ["simple", "basic", "plain"]) else 1
    else:
        score += 1

    return score


def height_weight_score(piece_type: str, height: int, weight: int, item_title: str) -> int:
    title = lower_text(item_title)
    score = 0

    if height >= 170:
        if any(x in title for x in ["maxi", "wide", "long", "straight"]):
            score += 2
    else:
        if any(x in title for x in ["cropped", "mini", "slim", "ankle"]):
            score += 2

    if weight >= 80:
        if any(x in title for x in ["straight", "wide", "wrap", "high waist", "flowy", "a-line", "aline", "relaxed"]):
            score += 4
        if any(x in title for x in ["skinny", "bodycon", "tight"]):
            score -= 3
    elif weight >= 75:
        if any(x in title for x in ["straight", "wide", "wrap", "high waist", "flowy"]):
            score += 2
    else:
        if any(x in title for x in ["fitted", "slim", "cropped", "tailored"]):
            score += 2

    if piece_type == "jumpsuit" and "belted" in title:
        score += 1

    return score


def strict_filter(data: pd.DataFrame, piece_type: str, color: str, occasion: str, style: str, weight: int) -> pd.DataFrame:
    out = data[data["piece_type"] == piece_type].copy()

    if color != "أي لون":
        out = out[out["color"] == color].copy()

    if occasion != "أي مناسبة":
        out = out[out["occasion"] == occasion].copy()

    if style not in ["أي ستايل", "أي نمط"]:
        style_filtered = out[out["style"] == style].copy()
        if not style_filtered.empty:
            out = style_filtered

    if weight >= 80 and piece_type not in ["bag", "shoes", "glasses", "jewelry", "scarf"]:
        out = out[out["is_curve_plus"]].copy()

    return out.reset_index(drop=True)

def get_relaxed_same_type_only(
    piece_type: str,
    color: str,
    occasion: str,
    style: str,
    weight: int,
) -> pd.DataFrame:
    out = DATA[DATA["piece_type"] == piece_type].copy()

    if weight >= 80 and piece_type not in ["bag", "shoes", "glasses", "jewelry", "scarf"]:
        plus_only = out[out["is_curve_plus"]].copy()
        if not plus_only.empty:
            out = plus_only

    if occasion != "أي مناسبة":
        occ_filtered = out[out["occasion"] == occasion].copy()
        if not occ_filtered.empty:
            out = occ_filtered

    if style not in ["أي ستايل", "أي نمط"]:
        style_filtered = out[out["style"] == style].copy()
        if not style_filtered.empty:
            out = style_filtered

    if color != "أي لون":

        color_filtered = out[out["color"] == color].copy()
        if not color_filtered.empty:
            return color_filtered.reset_index(drop=True)
        return pd.DataFrame(columns=out.columns)

    return out.reset_index(drop=True)

def relaxed_filter(data: pd.DataFrame, piece_type: str, color: str, occasion: str, style: str, weight: int) -> pd.DataFrame:
    out = data[data["piece_type"] == piece_type].copy()

    if weight >= 80 and piece_type not in ["bag", "shoes", "glasses", "jewelry", "scarf"]:
        plus_only = out[out["is_curve_plus"]].copy()
        if not plus_only.empty:
            out = plus_only

    if occasion != "أي مناسبة":
        occ_filtered = out[out["occasion"] == occasion].copy()
        if not occ_filtered.empty:
            out = occ_filtered

    if style not in ["أي ستايل", "أي نمط"]:
        style_filtered = out[out["style"] == style].copy()
        if not style_filtered.empty:
            out = style_filtered

    if color != "أي لون":
        
        color_filtered = out[out["color"] == color].copy()
        if not color_filtered.empty:
            return color_filtered.reset_index(drop=True)
        return pd.DataFrame(columns=out.columns)

    return out.reset_index(drop=True)


def rank_items(piece_type: str, color: str, occasion: str, style: str, height: int, weight: int, body_shape: str) -> pd.DataFrame:
    out = strict_filter(DATA, piece_type, color, occasion, style, weight).copy()

    if out.empty:
        out = get_relaxed_same_type_only(piece_type, color, occasion, style, weight).copy()

    
    out = out[out["piece_type"] == piece_type].copy()

    if out.empty:
        return out

    out["features"] = out.apply(build_features, axis=1)

    style_hints = []
    occasion_hints = []

    
    if occasion == "حفلة":
       occasion_hints.extend(["party", "formal", "elegant", "night", "dressy"])
    elif occasion == "عمل":
        occasion_hints.extend(["work", "office", "smart", "comfortable", "practical"])
    elif occasion == "يومي":
        occasion_hints.extend(["casual", "daily", "comfortable", "relaxed"])
    elif occasion == "سفر":
        occasion_hints.extend(["comfortable", "practical", "easy", "travel"])
        
    
    if style == "أنيق":
       style_hints.extend(["chic", "elegant", "feminine", "refined"])
    elif style == "كلاسيكي":
        style_hints.extend(["classic", "formal", "smart", "timeless"])
    elif style == "عصري":
        style_hints.extend(["modern", "clean", "fashion", "stylish"])
    elif style == "يومي":
        style_hints.extend(["casual", "relaxed", "easy"])
    elif style == "بسيط":
        style_hints.extend(["minimal", "clean", "simple"])

    user_profile_parts = [
        body_shape if piece_type not in ["bag", "shoes", "glasses", "jewelry", "scarf"] else "",
        occasion if occasion != "أي مناسبة" else "",
        style if style not in ["أي ستايل", "أي نمط"] else "",
        color if color != "أي لون" else "",
        piece_type,
        "curve plus" if (weight >= 80 and piece_type not in ["bag", "shoes", "glasses", "jewelry", "scarf"]) else "",
        "tall" if height >= 170 else "short",
        " ".join(occasion_hints),
        " ".join(style_hints),
    ]
    user_profile = " ".join([clean_text(x) for x in user_profile_parts if clean_text(x)])

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(out["features"].tolist() + [user_profile])

    ai_scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    out["ai_score"] = ai_scores

    if piece_type in ["bag", "shoes", "glasses", "jewelry", "scarf"]:
        out["rule_score"] = 1.0
        out["rule_score_norm"] = 1.0
    else:
        out["rule_score"] = out.apply(
            lambda r: body_shape_score(r["title"], piece_type, body_shape)
            + height_weight_score(piece_type, height, weight, r["title"])
            + (5 if (weight >= 80 and bool(r["is_curve_plus"])) else 0),
            axis=1,
        )

        max_rule = out["rule_score"].max()
        if max_rule > 0:
            out["rule_score_norm"] = out["rule_score"] / max_rule
        else:
            out["rule_score_norm"] = 0.0

    out["final_score"] = (out["ai_score"] * 0.7) + (out["rule_score_norm"] * 0.3)

    return out.sort_values("final_score", ascending=False).reset_index(drop=True)

def pick_one(df: pd.DataFrame, seed_value: str, pool_size: int = 15):
    if df.empty:
        return None

    rng = random.Random(seed_value)
    top_pool = df.head(min(pool_size, len(df))).copy()
    row = top_pool.iloc[rng.randrange(len(top_pool))]
    return row.to_dict()


def pick_harmonized_pair(top_df: pd.DataFrame, bottom_df: pd.DataFrame, seed_value: str):
    if top_df.empty:
        return None, None

    top_item = pick_one(top_df, seed_value + "|top")
    if not top_item:
        return None, None

    if bottom_df.empty:
        return top_item, None

    preferred_bottoms = bottom_df.copy()

    if top_item.get("color"):
        harmony_colors = get_harmony_colors(top_item["color"])
        if harmony_colors:
            filtered_bottoms = preferred_bottoms[preferred_bottoms["color"].isin(harmony_colors)].copy()
            if not filtered_bottoms.empty:
                preferred_bottoms = filtered_bottoms

        non_same_color = preferred_bottoms[preferred_bottoms["color"] != top_item["color"]].copy()
        if not non_same_color.empty:
            preferred_bottoms = non_same_color

    bottom_item = pick_one(preferred_bottoms, seed_value + "|bottom")

    if bottom_item is None:
        bottom_item = pick_one(bottom_df, seed_value + "|bottom_fallback")

    return top_item, bottom_item


def choose_best_by_color_priority(ranked_df: pd.DataFrame, priority_colors: list[str], seed_value: str):
    if ranked_df.empty:
        return None

    for color in priority_colors:
        same_color = ranked_df[ranked_df["color"] == color].copy()
        if not same_color.empty:
            return pick_one(same_color, seed_value + "|" + color, pool_size=10)

    return pick_one(ranked_df, seed_value + "|fallback", pool_size=10)


def pick_matching_accessory(
    piece_type: str,
    priority_colors: list[str],
    occasion: str,
    style: str,
    height: int,
    weight: int,
    body_shape: str,
    seed_value: str,
):
    ranked = rank_items(piece_type, "أي لون", occasion, style, height, weight, body_shape)
    ranked = ranked[ranked["piece_type"] == piece_type].copy()

    if ranked.empty:
        return None

    filtered = ranked[ranked["color"].isin(priority_colors)].copy()
    if not filtered.empty:
        ranked = filtered

    return choose_best_by_color_priority(ranked.head(15), priority_colors, seed_value)


def is_heavy_winter_accessory(title: str) -> bool:
    t = lower_text(title)
    return any(x in t for x in [
        "fuzzy", "fur", "faux fur", "wool", "knit", "thermal",
        "puffer", "quilted", "chunky", "thick", "plaid",
        "fleece", "sherpa", "cashmere", "ribbed knit"
    ])


def is_light_fancy_accessory(title: str) -> bool:
    t = lower_text(title)
    return any(x in t for x in [
        "lace", "satin", "sheer", "mesh", "silk",
        "opera", "evening", "bridal", "rhinestone", "pearl",
        "chiffon", "organza", "sequin", "glitter", "tulle"
    ])


def is_winter_main_item(title: str) -> bool:
    t = lower_text(title)
    return any(x in t for x in [
        "coat", "jacket", "hoodie", "sweater", "cardigan",
        "knit", "wool", "puffer", "long sleeve", "turtleneck",
        "high neck", "fleece", "thermal", "sherpa", "quilted",
        "fur", "faux fur", "chunky", "thick"
    ])


def is_winter_text(title: str) -> bool:
    t = lower_text(title)
    return any(x in t for x in [
        "coat", "jacket", "hoodie", "sweater", "cardigan",
        "knit", "wool", "puffer", "fur", "faux fur", "fuzzy",
        "fleece", "thermal", "turtleneck", "high neck", "long sleeve",
        "sherpa", "quilted", "chunky", "thick", "ribbed", "plaid"
    ])


def is_summer_text(title: str) -> bool:
    t = lower_text(title)
    return any(x in t for x in [
        "sleeveless", "tank", "short sleeve", "crop", "cropped",
        "linen", "mesh", "backless", "halter", "tube", "strapless",
        "camisole", "cami", "mini", "sandals", "sandal"
    ])


def is_formal_text(title: str) -> bool:
    t = lower_text(title)
    return any(x in t for x in [
        "satin", "sequin", "glitter", "lace", "velvet", "chiffon",
        "organza", "silk", "evening", "party", "dressy", "rhinestone",
        "pearl", "corset", "gown"
    ])


def is_heavy_winter_shoe(title: str) -> bool:
    t = lower_text(title)
    return any(x in t for x in [
        "fuzzy", "fur", "faux fur", "wool", "knit", "thermal",
        "puffer", "sherpa", "fleece", "snow", "winter"
    ])


def is_long_or_heavy_boot(title: str) -> bool:
    t = lower_text(title)
    return any(x in t for x in [
        "knee boot", "knee-high", "knee high", "over-the-knee",
        "over the knee", "thigh", "riding boot", "tall boot",
        "combat boot", "rain boot", "snow boot", "boots", " boot"
    ])


def is_light_or_formal_shoe(title: str) -> bool:
    t = lower_text(title)
    return any(x in t for x in [
        "heel", "heels", "pump", "pumps", "sandal", "sandals",
        "mule", "mules", "flat", "flats", "loafer", "loafers",
        "mary jane", "ballet", "slingback", "strappy", "pointed"
    ])


def get_main_items(result: dict) -> list[dict]:
    items = []
    for key in ["single", "top", "bottom"]:
        item = result.get(key)
        if item:
            items.append(item)
    return items


def look_context(result: dict, form: dict) -> dict:
    titles = " ".join([clean_text(item.get("title", "")) for item in get_main_items(result)])
    occasion = clean_text(form.get("occasion", ""))
    style = clean_text(form.get("style", ""))

    winter = is_winter_text(titles)
    summer = is_summer_text(titles) and not winter
    formal = (occasion == "حفلة") or (style in ["أنيق", "كلاسيكي"]) or is_formal_text(titles)

    return {
        "titles": titles,
        "is_winter": winter,
        "is_summer": summer,
        "is_formal": formal,
    }


def filter_shoes_by_look(shoes: pd.DataFrame, context: dict) -> pd.DataFrame:
    if shoes.empty:
        return shoes

    out = shoes.copy()
    if not context["is_winter"]:
        
        out = out[~out["title"].apply(is_heavy_winter_shoe)].copy()
        out = out[~out["title"].apply(is_long_or_heavy_boot)].copy()

    if context["is_summer"] or context["is_formal"]:
        preferred = out[out["title"].apply(is_light_or_formal_shoe)].copy()
        if not preferred.empty:
            out = preferred

    return out


def filter_scarves_by_look(scarves: pd.DataFrame, context: dict) -> pd.DataFrame:
    if scarves.empty:
        return scarves

    out = scarves.copy()
    out["is_heavy"] = out["title"].apply(is_heavy_winter_accessory)
    out["is_light_fancy"] = out["title"].apply(is_light_fancy_accessory)

    if context["is_winter"]:
        return out

    if context["is_formal"]:
        
        light = out[(out["is_heavy"] == False) | (out["is_light_fancy"] == True)].copy()
        return light

    
    return out[(out["is_heavy"] == False) & (out["is_light_fancy"] == True)].copy()

def add_accessories_to_result(result: dict, form: dict, seed_base: str):
    wants_any_accessory = any([
        form.get("include_bag"),
        form.get("include_shoes"),
        form.get("include_glasses"),
        form.get("include_jewelry"),
        form.get("include_scarf"),
    ])

    if not wants_any_accessory:
        return result

    has_main_look = bool(
        result.get("single")
        or result.get("top")
        or result.get("bottom")
    )

    if not has_main_look:
        result["bag"] = None
        result["shoes"] = None
        result["glasses"] = None
        result["jewelry"] = None
        result["scarf"] = None
        return result

    base_colors = []

    if result.get("top") and result["top"].get("color"):
        base_colors.append(result["top"]["color"])

    if result.get("bottom") and result["bottom"].get("color"):
        base_colors.append(result["bottom"]["color"])

    if result.get("single") and result["single"].get("color"):
        base_colors.append(result["single"]["color"])

    priority_colors = get_accessory_priority_colors(base_colors)
    context = look_context(result, form)

    chosen_bag = None
    chosen_shoes = None
    chosen_glasses = None
    chosen_jewelry = None
    chosen_scarf = None

    if form.get("include_bag"):
        chosen_bag = pick_matching_accessory(
            "bag",
            priority_colors,
            form["occasion"],
            form["style"],
            form["height"],
            form["weight"],
            form["body_shape"],
            seed_base + "|bag",
        )

    if form.get("include_shoes"):
        shoe_priority = priority_colors.copy()
        if chosen_bag and chosen_bag.get("color"):
            bag_color = chosen_bag["color"]
            shoe_priority = [bag_color] + [c for c in shoe_priority if c != bag_color]

        shoes_ranked = rank_items(
            "shoes",
            "أي لون",
            form["occasion"],
            form["style"],
            form["height"],
            form["weight"],
            form["body_shape"],
        )
        shoes_ranked = shoes_ranked[shoes_ranked["piece_type"] == "shoes"].copy()
        shoes_ranked = filter_shoes_by_look(shoes_ranked, context)

        color_filtered = shoes_ranked[shoes_ranked["color"].isin(shoe_priority)].copy()
        if not color_filtered.empty:
            shoes_ranked = color_filtered

        chosen_shoes = (
            choose_best_by_color_priority(
                shoes_ranked.head(15),
                shoe_priority,
                seed_base + "|shoes",
            )
            if not shoes_ranked.empty
            else None
        )

    if form.get("include_glasses"):
        chosen_glasses = pick_matching_accessory(
            "glasses",
            priority_colors,
            form["occasion"],
            form["style"],
            form["height"],
            form["weight"],
            form["body_shape"],
            seed_base + "|glasses",
        )

    if form.get("include_jewelry"):
        jewelry_priority = priority_colors.copy()
        if chosen_bag and chosen_bag.get("color"):
            bag_color = chosen_bag["color"]
            jewelry_priority = [bag_color] + [c for c in jewelry_priority if c != bag_color]

        chosen_jewelry = pick_matching_accessory(
            "jewelry",
            jewelry_priority,
            form["occasion"],
            form["style"],
            form["height"],
            form["weight"],
            form["body_shape"],
            seed_base + "|jewelry",
        )

    if form.get("include_scarf"):
        scarf_ranked = rank_items(
            "scarf",
            "أي لون",
            form["occasion"],
            form["style"],
            form["height"],
            form["weight"],
            form["body_shape"],
        )
        scarf_ranked = scarf_ranked[scarf_ranked["piece_type"] == "scarf"].copy()
        scarf_ranked = filter_scarves_by_look(scarf_ranked, context)

        color_filtered = scarf_ranked[scarf_ranked["color"].isin(priority_colors)].copy()
        if not color_filtered.empty:
            scarf_ranked = color_filtered

        chosen_scarf = (
            choose_best_by_color_priority(
                scarf_ranked.head(15),
                priority_colors,
                seed_base + "|scarf",
            )
            if not scarf_ranked.empty
            else None
        )

    result["bag"] = chosen_bag
    result["shoes"] = chosen_shoes
    result["glasses"] = chosen_glasses
    result["jewelry"] = chosen_jewelry
    result["scarf"] = chosen_scarf
    return result

def build_result(form: dict, regen: str):
    lang = form.get("lang", "ar")

    def msg(ar, en):
            return en if lang == "en" else ar

    seed_base = "|".join(str(v) for v in form.values()) + "|" + regen
    result = {
        "top": None,
        "bottom": None,
        "single": None,
        "bag": None,
        "shoes": None,
        "glasses": None,
        "jewelry": None,
        "scarf": None,
        "message": "",
    }

    if form["look_type"] == "بلوزة + بنطالون":
        tops = rank_items("top", form["top_color"], form["occasion"], form["style"], form["height"], form["weight"], form["body_shape"])
        pants = rank_items("pants", form["bottom_color"], form["occasion"], form["style"], form["height"], form["weight"], form["body_shape"])
        top_item, bottom_item = pick_harmonized_pair(tops, pants, seed_base + "|pants_look")
        result["top"] = top_item
        result["bottom"] = bottom_item

        if result["top"] is None and result["bottom"] is None:
            result["message"] = msg(
    "لم يتم العثور على إطلالة مطابقة لهذه الخيارات. جرّبي لونًا أو نمطًا آخر.",
    "No matching outfit was found for these options. Try another color or style."
)

    elif form["look_type"] == "بلوزة + تنورة":
        tops = rank_items("top", form["top_color"], form["occasion"], form["style"], form["height"], form["weight"], form["body_shape"])
        skirts = rank_items("skirt", form["bottom_color"], form["occasion"], form["style"], form["height"], form["weight"], form["body_shape"])
        top_item, bottom_item = pick_harmonized_pair(tops, skirts, seed_base + "|skirt_look")
        result["top"] = top_item
        result["bottom"] = bottom_item

        if result["top"] is None and result["bottom"] is None:
            result["message"] = msg(
    "لم يتم العثور على إطلالة مطابقة لهذه الخيارات. جرّبي لونًا أو نمطًا آخر.",
    "No matching outfit was found for these options. Try another color or style."
)

    elif form["look_type"] == "جمبسوت":
        items = rank_items("jumpsuit", form["single_color"], form["occasion"], form["style"], form["height"], form["weight"], form["body_shape"])
        result["single"] = pick_one(items, seed_base + "|jumpsuit")
        if result["single"] is None:
            result["message"] = msg(
    "لم يتم العثور على جمبسوت مطابق لهذه الخيارات. جرّبي لونًا أو نمطًا آخر.",
    "No matching jumpsuit was found for these options. Try another color or style."
)

    elif form["look_type"] == "فستان":
        items = rank_items("dress", form["single_color"], form["occasion"], form["style"], form["height"], form["weight"], form["body_shape"])
        result["single"] = pick_one(items, seed_base + "|dress")
        if result["single"] is None:
            result["message"] = msg(
    "لم يتم العثور على فستان مطابق لهذه الخيارات. جرّبي لونًا أو نمطًا آخر.",
    "No matching dress was found for these options. Try another color or style."
)

    elif form["look_type"] == "طقم متناسق":
        items = rank_items("set", form["single_color"], form["occasion"], form["style"], form["height"], form["weight"],
                           form["body_shape"])
        items = items[items["piece_type"] == "set"].copy()
        result["single"] = pick_one(items, seed_base + "|set")
        if result["single"] is None:
            result["message"] = msg(
    "لم يتم العثور على طقم متناسق مطابق لهذه الخيارات. جرّبي لونًا أو نمطًا آخر.",
    "No matching set was found for these options. Try another color or style."
)

    result = add_accessories_to_result(result, form, seed_base)
    return result


def default_form():
    return {
        "height": 160,
        "weight": 60,
        "body_shape": BODY_SHAPES[0],
        "occasion": OCCASIONS[0],
        "style": STYLES[0],
        "look_type": LOOK_TYPES[0],
        "top_color": "أي لون",
        "bottom_color": "أي لون",
        "single_color": "أي لون",
        "include_bag": False,
        "include_shoes": False,
        "include_glasses": False,
        "include_jewelry": False,
        "include_scarf": False,
        "lang": "ar",
    }


def render_experience():
    form = default_form()
    result = {}
    regen = request.form.get("regen", "0")

    if request.method == "POST":
        form["height"] = int(request.form.get("height", 160))
        form["weight"] = int(request.form.get("weight", 60))
        form["body_shape"] = request.form.get("body_shape", BODY_SHAPES[0])
        form["occasion"] = request.form.get("occasion", OCCASIONS[0])
        form["style"] = request.form.get("style", STYLES[0])
        form["look_type"] = request.form.get("look_type", LOOK_TYPES[0])
        form["top_color"] = request.form.get("top_color", "أي لون")
        form["bottom_color"] = request.form.get("bottom_color", "أي لون")
        form["single_color"] = request.form.get("single_color", "أي لون")
        form["include_bag"] = request.form.get("include_bag") == "1"
        form["include_shoes"] = request.form.get("include_shoes") == "1"
        form["include_glasses"] = request.form.get("include_glasses") == "1"
        form["include_jewelry"] = request.form.get("include_jewelry") == "1"
        form["include_scarf"] = request.form.get("include_scarf") == "1"
        form["lang"] = request.form.get("lang", "ar")
        result = build_result(form, regen)

    color_options = {
        "top": get_available_colors("top"),
        "pants": get_available_colors("pants"),
        "skirt": get_available_colors("skirt"),
        "jumpsuit": get_available_colors("jumpsuit"),
        "dress": get_available_colors("dress"),
        "set": get_available_colors("set"),
        "bag": get_available_colors("bag"),
        "shoes": get_available_colors("shoes"),
        "glasses": get_available_colors("glasses"),
        "jewelry": get_available_colors("jewelry"),
        "scarf": get_available_colors("scarf"),
    }

    color_labels = {}
    for options in color_options.values():
        for c in options:
            color_labels[c] = color_label_ar(c)
    color_labels["أي لون"] = "أي لون"

    return render_template(
        "experience.html",
        body_shapes=BODY_SHAPES,
        occasions=OCCASIONS,
        styles=STYLES,
        look_types=LOOK_TYPES,
        color_options=color_options,
        color_labels=color_labels,
        form=form,
        result=result,
        regen=regen,
    )


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/experience", methods=["GET", "POST"])
def experience():
    return render_experience()

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        rating = request.form.get("rating")
        notes = request.form.get("notes")

        file_exists = os.path.isfile("feedback.csv")

        with open("feedback.csv", "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["Date", "Rating",  "Notes"])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                rating,
                notes
            ])

        return render_template("thank_you.html")

    return render_template("feedback.html")


@app.route("/admin/feedbacks")
def admin_feedbacks():
    feedbacks = []

    if os.path.isfile("feedback.csv"):
        with open("feedback.csv", "r", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            feedbacks = list(reader)

    return render_template("admin_feedbacks.html", feedbacks=feedbacks)

@app.route('/regenerate-piece/<piece_type>')
def regenerate_piece(piece_type):
    data = load_all_data()
    selected_color = request.args.get("color", "أي لون")

    

    if piece_type == "shoes":
        items = data[data["source_file"].str.lower().str.contains("shoe|shoes|حذاء|شوز", na=False)]

    elif piece_type == "bag":
        items = data[data["source_file"].str.lower().str.contains("bag|bags|شنطة|شنط", na=False)]

    elif piece_type == "top":
        items = data[data["source_file"].str.lower().str.contains("top|tops|shirt|blouse|بلوزة", na=False)]

    elif piece_type == "bottom":
        items = data[data["source_file"].str.lower().str.contains("pants|skirt|bottom|بنطال|تنورة", na=False)]

    elif piece_type == "single":
        items = data[data["source_file"].str.lower().str.contains("dress|jumpsuit|set|فستان|جمبسوت|طقم", na=False)]

    elif piece_type == "glasses":
        items = data[data["source_file"].str.lower().str.contains("glasses|fashion-glasses", na=False)]

    elif piece_type == "jewelry":
        items = data[data["source_file"].str.lower().str.contains("jewelry", na=False)]

    elif piece_type == "scarf":
        items = data[data["source_file"].str.lower().str.contains("gloves|scarves|scarf", na=False)]

    else:
        return jsonify({"error": "Unknown piece type"}), 400

    if selected_color not in ["أي لون", "Any", "Any Color"] and "color" in items.columns:
        color_items = items[items["color"] == selected_color]
        if not color_items.empty:
            items = color_items
    
    if items.empty:
        return jsonify({"error": "No items"}), 404

    new_item = items.sample(1).iloc[0]

    return jsonify({
        "image_url": new_item["image_url"],
        "title": new_item["title"],
        "product_url": new_item["product_url"]
    })


@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)