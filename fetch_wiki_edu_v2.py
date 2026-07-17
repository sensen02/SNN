"""Extract quality Wikipedia text for low-difficulty education subjects.
Filters by TEXT QUALITY first, uses title only for categorization.
"""
import pyarrow.parquet as pq
import json, os, re

SRC_DIR = "/tmp/opencode/wiki"
DST = os.path.expanduser("~/srcn_v2_balanced/annotated_corpus_v2.jsonl")

# === Topic keywords (title match, for categorization only) ===
CATEGORIES = [
    ("数学", r"数学|算术|代数|几何|微积分|概率|统计|数论|函数|方程|圆周率|勾股|三角函数|矩阵|集合"),
    ("物理", r"物理|力学|牛顿|引力|电磁|电场|磁场|相对论|量子|热力学|光学|声学|能量|动量|波动|折射|原子结构"),
    ("化学", r"化学|元素周期|化合物|化学键|氧化|还原|酸碱性|催化剂|溶液|浓度|分子结构|电解|电离|同位素"),
    ("生物", r"生物|细胞|DNA|基因|进化|光合作用|生态系统|食物链|动物|植物|微生物|昆虫|哺乳|鸟纲|鱼类|爬行"),
    ("地理", r"地理|地球|板块|火山|地震|气候|大气层|海洋|河流|山脉|沙漠|冰川|赤道|南极|北极"),
    ("天文", r"天文|宇宙|恒星|行星|太阳系|银河|黑洞|月球|火星|木星|土星|超新星|星云|星座|日食|月食"),
    ("语文", r"汉字|文言文|古诗|唐诗|宋词|元曲|成语|谚语|修辞|比喻|拟人|语法|主语|谓语|宾语|诗经|论语|孟子|庄子|李白|杜甫|苏轼|鲁迅|红楼梦|三国演义|西游记|水浒|标点|拼音"),
    ("历史", r"古代|朝代|四大发明|造纸|印刷|火药|指南针|丝绸之路|科举|甲骨文|青铜|秦朝|汉朝|唐朝|宋朝|明朝|清朝|文艺复兴|工业革命|启蒙运动"),
    ("科学", r"科学方法|实验|观测|定理|定律|计量|单位|逻辑推理|假设|分类|归纳|演绎"),
]

# === Political exclusion ===
EXCLUDE_TITLE = [
    r"政治", r"政党", r"选举", r"总统", r"总理", r"主席", r"总书记", r"政治局",
    r"人大", r"政协", r"中共", r"共产党", r"国民党", r"文革", r"天安门",
    r"抗议", r"游行", r"镇压", r"革命", r"政变", r"战争", r"战役",
    r"军队", r"导弹", r"核武器", r"间谍", r"公安", r"国家安全",
    r"领土争端", r"主权", r"独立", r"分裂", r"台湾问题",
    r"国共", r"抗日", r"解放战争", r"建国",
]
EXCLUDE_TEXT = EXCLUDE_TITLE  # reuse for text-level filtering

def is_political(text):
    return any(re.search(p, text) for p in EXCLUDE_TEXT)

# === Text quality checks ===
NOISE_LINES = re.compile(
    r'^(={2,}.*?={2,})$|'              # section headers: === xxx ===
    r'^(参考文献|参考资料|外部链接|相关条目|參考資料|外部連結|相關條目|'
    r'参阅|参见|延伸阅读|注釋|注释|脚注|来源|出处)$|'
    r'^[\[\(]?\d+[\]\)]?\s*$|'        # just numbers
    r'^\s*$'                            # blank lines
)

def text_quality(text):
    """Return (clean_text, quality_score 0-1)"""
    lines = text.split('\n')
    clean_lines = []
    noise_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            noise_count += 1
            continue
        if NOISE_LINES.match(line):
            noise_count += 1
            continue
        # Remove [[Category:xxx]], [[File:xxx]], etc.
        line = re.sub(r'\[\[(?:Category|File|Image|Media):[^\]]+\]\]', '', line, flags=re.I)
        line = re.sub(r'https?://\S+', '', line)
        if len(line) < 3:
            noise_count += 1
            continue
        clean_lines.append(line)
    total = len(lines) or 1
    quality = 1.0 - noise_count / total
    return '\n'.join(clean_lines), quality

def categorize(title):
    for cat_name, pat in CATEGORIES:
        if re.search(pat, title):
            return f"wikipedia_v1/百科知识/{cat_name}"
    return "wikipedia_v1/百科知识"

# === Main ===
out_f = open(DST, "w", encoding="utf-8")
total = 0
kept = 0
bad_short = 0
bad_quality = 0
bad_political = 0

for fn in sorted(os.listdir(SRC_DIR)):
    if not fn.endswith(".parquet"):
        continue
    fp = os.path.join(SRC_DIR, fn)
    print(f"\nProcessing {fn}...")
    pf = pq.ParquetFile(fp)
    for batch in pf.iter_batches(batch_size=5000):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            total += 1
            title = str(row.get("title", ""))
            text = str(row.get("text", ""))
            
            # Quick skip: very short articles
            if len(text) < 200:
                bad_short += 1
                continue
            if is_political(title) or is_political(text[:1000]):
                bad_political += 1
                continue
            
            cleaned, quality = text_quality(text)
            if len(cleaned) < 150 or quality < 0.3:
                bad_quality += 1
                continue
            
            meta = categorize(title)
            out_f.write(json.dumps({
                "text": f"meta_tag:{meta}\n{cleaned}"
            }, ensure_ascii=False) + "\n")
            kept += 1
        
        if total % 50000 == 0:
            print(f"  {kept} kept / {total} scanned | short:{bad_short} low_qual:{bad_quality} political:{bad_political}", flush=True)
    print(f"  {fn} done: {kept} kept / {total} total | short:{bad_short} low_qual:{bad_quality} political:{bad_political}")

out_f.close()
print(f"\nDone: {kept} kept (short:{bad_short} low_qual:{bad_quality} political:{bad_political}) total={total}")
print(f"Output: {DST}")
print(f"Lines: {sum(1 for _ in open(DST))}")
