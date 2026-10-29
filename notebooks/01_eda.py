# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: datamate (py3.12)
#     language: python
#     name: datamate
# ---

# %% [markdown]
# # 01 — EDA: DataMate
#
# Mục tiêu: tìm pattern tổng thể của câu hỏi, đáp án, dữ liệu shop và rulebook, để chọn luồng xử lý
# và biết train có đại diện cho public test hay không.
#
# 1. Câu hỏi: số lượng, độ "bẩn" ngôn ngữ, train vs public
# 2. Chủ đề và biểu thức thời gian của câu hỏi, so với type đáp án
# 3. Đáp án (train)
# 4. Shop: ngành, file, sheet, kích thước
# 5. Header và độ phủ README
# 6. Độ bẩn giá trị ô (tiền, ngày, số lượng)
# 7. Trạng thái đơn, khách lẻ, ô trống theo nhóm, dòng trùng
# 8. Rulebook: train vs public
# 9. `as_of_date` so với khoảng dữ liệu

# %%
import pickle
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from datamate.io import (DATA, business_ids, load_answers, load_questions, parse_readme, read_raw_tables,
                         split_dir)
from datamate.profile import cell_kind, find_header_row, readme_columns_for, role_columns, with_header
from datamate.text import fold, has_diacritics

pd.set_option("display.max_columns", 40, "display.width", 200, "display.max_colwidth", 80)
SPLITS = ["train", "public_test"]

# %% [markdown]
# ## 0. Nạp dữ liệu (cache raw tables)
# openpyxl đọc 44 shop mất vài phút, nên lần đầu đọc rồi lưu pickle vào `data/cache/`.

# %%
CACHE = DATA / "cache"
CACHE.mkdir(exist_ok=True)


def load_split_tables(split):
    path = CACHE / f"raw_{split}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    tabs = {b: read_raw_tables(split_dir(split) / "data" / b) for b in business_ids(split)}
    path.write_bytes(pickle.dumps(tabs))
    return tabs


TABLES = {s: load_split_tables(s) for s in SPLITS}
READMES = {s: {b: parse_readme(split_dir(s) / "data" / b / "README.md") for b in business_ids(s)} for s in SPLITS}
Q = pd.concat([load_questions(s).assign(split=s) for s in SPLITS], ignore_index=True)
A = load_answers("train")
QA = Q[Q.split == "train"].merge(A, on="id")
print({s: len(TABLES[s]) for s in SPLITS}, "businesses;", len(Q), "questions")

# %% [markdown]
# ## 1. Câu hỏi: ngôn ngữ

# %%
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
TEEN = re.compile(r"\b(dthu|bn|bnhiu|bnh|nhiu|nhiêu|z|zậy|ko|k|hok|mik|mk|kua|kó|cko|jùm|nhứt|th\d|t\d+|đc|dc|sp|ad|e ơi|o đờ|bill)\b", re.I)

Q["len"] = Q.question.str.len()
Q["no_diacritics"] = ~Q.question.map(has_diacritics)
Q["emoji"] = Q.question.str.contains(EMOJI)
Q["teencode"] = Q.question.str.contains(TEEN)
Q.groupby("split")[["len", "no_diacritics", "emoji", "teencode"]].mean().round(3)

# %%
for s in SPLITS:
    print(f"--- {s}: 8 câu ngẫu nhiên")
    print(Q[Q.split == s].sample(8, random_state=0)[["business_id", "as_of_date", "question"]].to_string(index=False))

# %% [markdown]
# ## 2. Chủ đề và biểu thức thời gian
# Gắn nhãn thô bằng regex trên câu đã bỏ dấu (`fold`). Một câu có thể có nhiều nhãn.
# Đây là công cụ EDA, không phải bộ phân loại để dùng khi chạy thật.

# %%
TOPIC = {
    "revenue": r"doanh thu|dthu|doanh so|tien ban|tien hang|thu ve|ban (duoc |dc )?(bao nhieu|bn|bnhiu|nhieu|may) tien|nhieu tien nhat",
    "orders": r"\bdon\b|o do|\bbill\b|hoa don",
    "cancel": r"\bhuy\b",
    "return": r"tra hang|tra lai|doi tra",
    "debt": r"\bno\b|cong no|chua tra|thieu tien|tra gop|gop",
    "inventory": r"\bton\b|het hang|con hang|con lai",
    "purchase": r"\bnhap\b|nha cung cap|\bncc\b",
    "profit_cost": r"\blai\b|\bloi\b|gia von|chi phi",
    "price": r"\bgia\b",
    "customer": r"\bkhach\b",
    "staff": r"nhan vien|\btho\b|\bnv\b",
    "item_rank": r"mat hang|san pham|\bsp\b|\bmon\b|\bhang\b",
    "payment": r"tien mat|chuyen khoan|\bcod\b|vi dien tu|hinh thuc|kenh|zalo|shopee|tiktok|grab",
    "expiry_batch": r"han (dung|su dung)|het han|\blo\b",
    "delivery": r"giao|cho di|\bdot\b",
    "warranty": r"bao hanh",
    "domain_specific": r"topping|size|\(m\)|\(l\)|voucher|diem|bim|sua bot|hoat chat|ke don|bien so|luot xe|imei|may cu|dong may|hoa hong",
}
SHAPE = {
    "pct": r"%|phan tram|ti le|ty le|ti trong|ty trong|chiem",
    "compare": r"so voi|hon hay kem|tang hay giam|len hay xuong|chenh|cung ky|thay doi",
    "rank": r"nhat\b|nhứt|\btop\s?\d|xep|cao xuong thap|liet ke",
    "count_cond": r"(tu|it nhat|hon|tren|qua|duoi)\s?\d",
}
TIME = {
    "today": r"hom nay|hnay|ngay nay",
    "yesterday": r"hom qua",
    "this_week": r"tuan nay",
    "last_week": r"tuan (truoc|roi|vua roi)",
    "this_month": r"thang nay|\bth nay",
    "last_month": r"thang (truoc|roi)",
    "last_n_days": r"\d+ ngay (gan|qua|vua)",
    "ytd": r"tu dau nam|dau nam (toi|den)",
    "quarter": r"\bquy\b|\bq[1-4]\b",
    "tet": r"\btet\b",
    "same_period_ly": r"cung ky",
    "explicit_month": r"(thang|\bth|\bt)\s?\d{1,2}\s?(/|nam|-)\s?\d{2,4}|\b\d{1,2}/\d{2,4}\b(?!/)",
    "explicit_date": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|ngay \d{1,2}",
    "until_date": r"(den|toi) het (ngay|hom)|tinh (den|toi)",
    "shift": r"ca (sang|chieu)",
    "weekday": r"thu (hai|ba|tu|nam|sau|bay|2|3|4|5|6|7)|chu nhat",
}


def tag(texts, table):
    f = texts.map(fold)
    return pd.DataFrame({k: f.str.contains(v, regex=True) for k, v in table.items()})


Qf = Q.question
tags = pd.concat({"topic": tag(Qf, TOPIC), "shape": tag(Qf, SHAPE), "time": tag(Qf, TIME)}, axis=1)
by_split = tags.groupby(Q.split).mean().T.round(3)
by_split["diff_pub_minus_train"] = (by_split.public_test - by_split.train).round(3)
by_split.sort_values("diff_pub_minus_train")

# %%
print("Câu không bắt được biểu thức thời gian nào:")
no_time = ~tags["time"].any(axis=1)
print(no_time.groupby(Q.split).mean().round(3).to_dict())
print(Q[no_time].sample(12, random_state=1)[["split", "question"]].to_string(index=False))

# %% [markdown]
# ### Type đáp án theo shape (train)

# %%
tr_tags = tags[Q.split == "train"].set_index(Q[Q.split == "train"].id)
qa_t = QA.set_index("id").join(tr_tags["shape"])
pd.crosstab(qa_t["type"], [qa_t["pct"], qa_t["rank"]], margins=True)

# %% [markdown]
# ## 3. Đáp án (train)

# %%
print(A.type.value_counts().to_frame("n").assign(share=lambda d: (d.n / d.n.sum()).round(3)).to_string())
per_biz = pd.crosstab(QA.business_id, QA.type)
per_biz.describe().round(1)

# %%
num = QA[QA.type == "number"].answer.astype(float)
print("number: integer share", (num % 1 == 0).mean().round(3), "| negative", (num < 0).mean().round(3),
      "| median", num.median(), "| >=1e6 share", (num.abs() >= 1e6).mean().round(3))
pct = QA[QA.type == "percent"].answer.astype(float)
print("percent: negative share", (pct < 0).mean().round(3), "| >100 share", (pct.abs() > 100).mean().round(3))
dates = QA[QA.type == "date"].answer
print("date: month-only share", dates.str.len().eq(7).mean().round(3))
lst = QA[QA.type == "list"].answer
print("list length:", Counter(map(len, lst)).most_common(), "| numeric lists",
      round(lst.map(lambda x: all(isinstance(v, (int, float)) for v in x)).mean(), 3))
print("alias non-empty by type:", QA[QA.alias.map(len) > 0].type.value_counts().to_dict())

# %%
print("Câu unanswerable trong train:")
print(QA[QA.type == "unanswerable"][["business_id", "as_of_date", "question"]].to_string(index=False))

# %%
print("Table answers:")
for _, r in QA[QA.type == "table"].head(4).iterrows():
    print(r.question, "->", r.answer["columns"], r.answer["rows"][:2])

# %% [markdown]
# ## 4. Shop: ngành, file, kích thước
# Ngành đoán từ từ khoá trong README cộng câu hỏi của shop, chỉ để nhóm shop khi so sánh.
# Thứ tự quan trọng: "điện thoại" phải đứng sau vì cột "số điện thoại" và kênh "qua điện thoại" có ở mọi ngành.

# %%
INDUSTRY = [
    ("nhà thuốc", r"hoat chat|thuoc ke don"),
    ("sửa xe", r"sua xe|luot xe"),
    ("mẹ và bé", r"me va be|\bbim\b|sua bot"),
    ("điện thoại", r"imei|dong may|tra gop|may cu|cua hang dien thoai"),
    ("VLXD", r"xi mang|dot giao|giao nhieu dot"),
    ("quán ăn/nhậu", r"phong lanh|khu vuc cua ban|so khach cua don"),
    ("cà phê/trà sữa", r"topping|size [ml]\b|\((m|l)\)"),
]


def industry(text):
    f = fold(text)
    return next((name for name, rx in INDUSTRY if re.search(rx, f)), "bán lẻ/tạp hoá")


rows = []
for s in SPLITS:
    for b, tabs in TABLES[s].items():
        rd = READMES[s][b]
        qtext = " ".join(Q[Q.business_id == b].question)
        rows.append(dict(split=s, biz=b, industry=industry(rd.text + " " + qtext),
                         col_docs=len(rd.columns), files=len({t.file for t in tabs}),
                         tables=len(tabs), rows=sum(len(t.df) for t in tabs),
                         formats=",".join(sorted({t.file.rsplit(".", 1)[-1] for t in tabs})),
                         last_date=rd.last_date, n_q=(Q.business_id == b).sum(),
                         kind=rd.text.split("\n")[2][:60]))
BIZ = pd.DataFrame(rows)
print("Shop có README không mô tả cột nào (chỉ có Quy ước):",
      BIZ[BIZ.col_docs == 0].groupby("split").biz.apply(list).to_dict())
pd.crosstab(BIZ.industry, BIZ.split)

# %% [markdown]
# **Lưu ý:** nhãn ngành ở trên chỉ gần đúng (vd. dev_004 là VLXD nhưng regex gán "điện thoại").
# Đã kiểm tay bằng README: train **không** có shop điện thoại, sửa xe, mẹ và bé (rulebook train chỉ có mục
# Cà phê, Nhà thuốc, VLXD). Public có 4 shop ngành mới: dev_011, dev_014 (điện thoại), dev_012 (sửa xe),
# dev_013 (mẹ và bé) = 210/1000 câu.

# %%
print(BIZ.sort_values(["split", "industry"]).to_string(index=False))

# %% [markdown]
# ## 5. Header và độ phủ README
# Với mỗi bảng: tìm dòng header khớp nhiều tên cột README nhất trong 10 dòng đầu.

# %%
hrows = []
for s in SPLITS:
    for b, tabs in TABLES[s].items():
        for t in tabs:
            cols = readme_columns_for(READMES[s][b], t)
            row, cov = find_header_row(t.df, list(cols))
            hrows.append(dict(split=s, biz=b, file=t.file, sheet=t.sheet, n_rows=len(t.df), n_cols=t.df.shape[1],
                              readme_cols=len(cols), header_row=row, coverage=cov))
HDR = pd.DataFrame(hrows)
# Mô tả khớp tên file nhưng không khớp cột nào (vd. sheet phụ `Ghi tắt` của dev_011) thì coi như không có mô tả.
HDR.loc[HDR.coverage == 0, "readme_cols"] = 0
print("Bảng không có mô tả cột trong README:", (HDR.readme_cols == 0).sum(), "/", len(HDR))
print("Phân bố dòng header:", HDR[HDR.readme_cols > 0].header_row.value_counts().to_dict())
print("Độ phủ cột README (bảng có mô tả):")
print(HDR[HDR.readme_cols > 0].coverage.describe().round(3).to_string())
HDR[(HDR.readme_cols > 0) & (HDR.coverage < 1)].head(20)

# %%
print("Bảng README không mô tả (thường là sheet phụ):")
print(HDR[HDR.readme_cols == 0][["split", "biz", "file", "sheet", "n_rows", "n_cols"]].head(25).to_string(index=False))

# %% [markdown]
# ## 6. Độ bẩn giá trị ô
# Đếm kiểu giá trị (tiền `55k`, `2,825tr`, `55.000đ`, số kèm đơn vị, ngày `1/4`, ...) trên mọi ô dưới header.

# %%
krows = []
for s in SPLITS:
    for b, tabs in TABLES[s].items():
        hdr = HDR[(HDR.split == s) & (HDR.biz == b)]
        cnt = Counter()
        for t, (_, h) in zip(tabs, hdr.iterrows()):
            body = t.df.iloc[h.header_row + 1:]
            for col in body.columns:
                cnt.update(body[col].map(cell_kind).tolist())
        krows.append(dict(split=s, biz=b, **cnt))
KIND = pd.DataFrame(krows).fillna(0).set_index(["split", "biz"])
MESSY = ["money_dong", "money_k", "money_tr", "money_word", "thousands_dot", "thousands_comma", "num_with_unit",
         "date_dmy", "date_dmy_dash", "date_dm", "date_thg", "date_iso", "datetime_obj", "time_hm"]
messy = (KIND.reindex(columns=MESSY, fill_value=0) > 0).astype(int)
print("Số shop có ít nhất một ô kiểu đó:")
print(messy.groupby(level="split").sum().T.to_string())

# %%
fig, ax = plt.subplots(figsize=(10, 9))
ax.imshow(messy.values, aspect="auto", cmap="Greys")
ax.set_xticks(range(len(MESSY)), MESSY, rotation=60, ha="right")
ax.set_yticks(range(len(messy)), [b for _, b in messy.index], fontsize=7)
ax.set_title("Kiểu giá trị xuất hiện ở mỗi shop")
plt.tight_layout()

# %%
def sample_cells(split, biz, kind, k=8):
    hdr = HDR[(HDR.split == split) & (HDR.biz == biz)]
    out = []
    for t, (_, h) in zip(TABLES[split][biz], hdr.iterrows()):
        body = t.df.iloc[h.header_row + 1:]
        for col in body.columns:
            vals = body[col][body[col].map(cell_kind) == kind]
            out += vals.astype(str).head(3).tolist()
    return out[:k]


for kind in ["money_k", "money_tr", "money_dong", "num_with_unit", "date_dm", "date_thg"]:
    ex = next(((s, b) for (s, b), v in messy[kind].items() if v), None)
    if ex:
        print(f"{kind:14s} {ex[1]}: {sample_cells(*ex, kind)}")

# %% [markdown]
# ## 7. Trạng thái đơn, khách lẻ, ô trống theo nhóm, dòng trùng

# %%
def framed(split, biz):
    """(table, header-applied df, README column map) for every described table of a shop."""
    hdr = HDR[(HDR.split == split) & (HDR.biz == biz)]
    for t, (_, h) in zip(TABLES[split][biz], hdr.iterrows()):
        cols = readme_columns_for(READMES[split][biz], t)
        if cols:
            yield t, with_header(t.df, h.header_row), cols


status, walkin, frows = Counter(), Counter(), []
for s in SPLITS:
    for b in TABLES[s]:
        for t, df, cols in framed(s, b):
            for c in role_columns(cols, "trạng thái đơn"):
                if c in df:
                    status.update((s, fold(v)) for v in df[c].dropna().astype(str))
            for c in role_columns(cols, "tên khách"):
                if c in df:
                    vc = df[c].dropna().astype(str).map(fold).value_counts()
                    walkin.update({(s, k): v for k, v in vc.head(3).items() if len(k) <= 12})
            for c in role_columns(cols, "mã đơn"):
                if c in df and len(df) > 0:
                    blank = df[c].isna().mean()
                    dup = df.duplicated(keep="first").mean()
                    frows.append(dict(split=s, biz=b, file=t.file, sheet=t.sheet, rows=len(df),
                                      order_id_blank=round(blank, 3), exact_dup_rows=round(dup, 4)))

st = pd.Series(status).unstack(0).fillna(0).astype(int).sort_values("train", ascending=False)
print("Giá trị trạng thái đơn (đã bỏ dấu):", len(st))
print(st.to_string())

# %%
print("Tên khách xuất hiện nhiều nhất và ngắn (ứng viên 'khách lẻ'):")
print(pd.Series(walkin).unstack(0).fillna(0).astype(int).sort_values("train", ascending=False).head(25).to_string())

# %%
FILL = pd.DataFrame(frows)
print("Bảng có ô mã đơn để trống (ghi ở dòng đầu đơn):", (FILL.order_id_blank > 0).sum(), "/", len(FILL))
print("Bảng có dòng trùng hệt nhau:", (FILL.exact_dup_rows > 0).sum(), "/", len(FILL))
FILL.sort_values("order_id_blank", ascending=False).head(15)

# %% [markdown]
# ## 8. Rulebook: train vs public

# %%
def rulebook_rules(split):
    text = (split_dir(split) / "rulebook.md").read_text(encoding="utf-8")
    rules, section = {}, None
    for line in text.splitlines():
        if line.startswith("#"):
            section = line.lstrip("# ").strip()
        m = re.match(r"^\| ([^|]+?) \| (.+) \|$", line)
        if m and not set(m.group(1)) <= set("-") and m.group(1) not in ("Luật", "Cách nói", "Cách ghi", "Từ", "type"):
            rules[(section, m.group(1))] = m.group(2)
    return rules


R = {s: rulebook_rules(s) for s in SPLITS}
added = sorted(set(R["public_test"]) - set(R["train"]))
removed = sorted(set(R["train"]) - set(R["public_test"]))
changed = sorted(k for k in set(R["train"]) & set(R["public_test"]) if R["train"][k] != R["public_test"][k])
print(f"rules: train={len(R['train'])} public={len(R['public_test'])} | added={len(added)} removed={len(removed)} changed={len(changed)}")
for name, keys in [("ADDED", added), ("REMOVED", removed), ("CHANGED", changed)]:
    print(f"\n## {name}")
    for sec, rule in keys:
        print(f"- [{sec}] {rule}")

# %% [markdown]
# ## 9. `as_of_date` so với khoảng dữ liệu

# %%
last = {b: pd.to_datetime(r.last_date, dayfirst=True) for s in SPLITS for b, r in READMES[s].items()}
Q["as_of"] = pd.to_datetime(Q.as_of_date)
Q["days_before_end"] = (Q.business_id.map(last) - Q.as_of).dt.days
print(Q.groupby("split").days_before_end.describe().round(1).to_string())
print("as_of == ngày cuối dữ liệu:", Q.groupby("split").days_before_end.apply(lambda d: (d == 0).mean().round(3)).to_dict())
print("as_of sau ngày cuối dữ liệu:", Q.groupby("split").days_before_end.apply(lambda d: (d < 0).mean().round(3)).to_dict())

# %%
print("Số as_of_date khác nhau mỗi shop:", Q.groupby(["split", "business_id"]).as_of_date.nunique().groupby("split").describe().round(1).to_string())

# %% [markdown]
# ## 10. Ngày đầu dữ liệu và câu unanswerable
# README chỉ ghi ngày cuối. Ngày đầu lấy từ mọi cột ngày phát sinh (bỏ ngày giao, hạn, tạo, đăng ký, nhập...),
# min các ô parse được; ô `1/4` không năm bỏ qua. Ô ISO parse riêng, không dùng `dayfirst` (sẽ đảo ngày/tháng).
# Giả thuyết (rulebook §7): kỳ hỏi nằm **hoàn toàn** ngoài khoảng dữ liệu thì unanswerable.
#
# **Kết quả:** heuristic dưới đây còn bỏ sót ngày đầu của nhiều shop (NaT), nên con số 7/70 là cận dưới.
# Kiểm tay: biz_121 có dữ liệu từ 03/2025, biz_106 từ 02/2025, biz_103 từ 04/2025 (sheet đầu `04.2025`),
# khớp với các câu unanswerable hỏi tháng 1–3/2025 của chính các shop đó.
# Phân loại tay 70 câu unanswerable của train: khoảng 1/3 là kỳ ngoài dữ liệu (kể cả "so với tháng liền trước"
# ở tháng đầu), khoảng 1/2 là shop không ghi loại thông tin đó (địa chỉ/đường, giới tính, tồn kho, nhập hàng,
# công nợ, kênh bán, hình thức thanh toán, lý do huỷ/trả), khoảng 1/10 là câu chủ quan hoặc cần kiến thức
# ngoài (tay nghề, lễ phép, liều dùng thuốc).

# %%
DATE_HEAD = re.compile(r"ngay|thoi gian|thoi diem|date")
NOT_SALE_DATE = re.compile(r"giao|han|tao|dang ky|nhap|tra|sinh|vao lam|ap dung|het")


def parse_date_cell(v):
    k = cell_kind(v)
    if k == "datetime_obj":
        return pd.Timestamp(v)
    if k in ("date_dmy", "date_dmy_dash"):
        return pd.to_datetime(str(v).strip(), dayfirst=True, errors="coerce")
    if k == "date_iso":
        return pd.to_datetime(str(v).strip()[:10], errors="coerce")
    if k == "date_thg":
        m = re.match(r"(\d{1,2}) thg (\d{1,2}) (\d{4})", str(v).strip())
        return pd.Timestamp(int(m[3]), int(m[2]), int(m[1])) if m else pd.NaT
    return pd.NaT


first = {}
for s in SPLITS:
    hdr_s = HDR[HDR.split == s]
    for b, tabs in TABLES[s].items():
        mins = []
        for t, (_, h) in zip(tabs, hdr_s[hdr_s.biz == b].iterrows()):
            df = with_header(t.df, h.header_row)
            for c in df.columns:
                fc = fold(c)
                if DATE_HEAD.search(fc) and not NOT_SALE_DATE.search(fc):
                    d = df[c].head(5000).map(parse_date_cell).dropna()
                    if len(d):
                        mins.append(d.min())
        first[b] = min(mins) if mins else pd.NaT
BIZ["first_date"] = BIZ.biz.map(first)
BIZ["last"] = pd.to_datetime(BIZ.last_date, dayfirst=True)
BIZ["months"] = ((BIZ["last"] - BIZ.first_date).dt.days / 30.4).round(1)
print(BIZ[["split", "biz", "industry", "first_date", "last_date", "months"]].to_string(index=False))

# %%
MONTH_RX = re.compile(r"(?:thang|\bth|\bt)\s?(\d{1,2})\s?(?:/|nam|-)\s?(20\d{2}|\d{2})\b")


def asked_month(q):
    m = MONTH_RX.search(fold(q))
    if not m:
        return pd.NaT
    y = int(m[2]) + (2000 if len(m[2]) == 2 else 0)
    return pd.Timestamp(y, int(m[1]), 1) if 1 <= int(m[1]) <= 12 else pd.NaT


qa = QA.assign(month=QA.question.map(asked_month), first=QA.business_id.map(first))
qa["month_end"] = qa.month + pd.offsets.MonthEnd(0)
qa["before_data"] = qa.month_end < qa["first"]
print("Câu có tháng cụ thể:", qa.month.notna().sum(), "| trong đó tháng nằm hẳn trước ngày đầu dữ liệu:", qa.before_data.sum())
print(pd.crosstab(qa.before_data, qa.type == "unanswerable", rownames=["before_data"], colnames=["unanswerable"]))
un = qa[qa.type == "unanswerable"]
print(f"\nUnanswerable giải thích được bằng 'kỳ trước dữ liệu': {un.before_data.sum()} / {len(un)}")
print(un[~un.before_data][["business_id", "question"]].to_string(index=False))

# %%
pub = Q[Q.split == "public_test"].assign(month=lambda d: d.question.map(asked_month),
                                         first=lambda d: d.business_id.map(first))
pub["before_data"] = (pub.month + pd.offsets.MonthEnd(0)) < pub["first"]
print("Public: câu hỏi về tháng nằm hẳn trước dữ liệu:", int(pub.before_data.sum()), "/", len(pub))
