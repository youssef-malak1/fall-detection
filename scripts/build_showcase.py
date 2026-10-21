"""Build a single self-contained HTML showcase of the whole project.

    python -m scripts.build_showcase

Images are embedded as base64 so the file can be mailed or opened from anywhere.
The numbers below are mirrored from the reports in ``reports/``; if you re-run the
evaluation, update them there first and then here.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES = PROJECT_ROOT / "reports" / "figures"
OUT = PROJECT_ROOT / "reports" / "showcase.html"

TEST_COUNT = 44

FIG = {
    "contact": "eda/adl-30-cam0_sheet.jpg",
    "pose_check": "eda/fall-20-cam0_pose_check.jpg",
    "fall23": "eda/fall-23-cam0_contact.jpg",
    "adl33": "eda/adl-33-cam0_contact.jpg",
    "cm_rule": "confusion_rule.png",
    "cm_logreg": "confusion_logreg.png",
    "cm_rf": "confusion_rf.png",
    "cm_temporal": "confusion_temporal.png",
    "tl_fall21": "temporal_timeline_fall-21-cam0.png",
    "tl_fall23": "temporal_timeline_fall-23-cam0.png",
    "tl_adl30": "temporal_timeline_adl-30-cam0.png",
    "tl_adl33": "temporal_timeline_adl-33-cam0.png",
    "inf_fall": "inference_fall-01-cam0.png",
    "inf_adl": "inference_adl-30-cam0.png",
}


def img(key: str, caption: str, width: str = "100%") -> str:
    path = FIGURES / FIG[key]
    if not path.exists():
        return f"<p class='missing'>missing figure: {html.escape(str(path))}</p>"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    ext = path.suffix.lstrip(".")
    return (
        f"<figure><img src='data:image/{ext};base64,{data}' style='width:{width}'/>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


def card(value: str, label: str, tone: str = "") -> str:
    return (
        f"<div class='card {tone}'><div class='value'>{value}</div>"
        f"<div class='label'>{label}</div></div>"
    )


def table(headers: list[str], rows: list[list[str]], highlight: int | None = None) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for i, row in enumerate(rows):
        cls = " class='hl'" if i == highlight else ""
        body.append("<tr" + cls + ">" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


HEAD = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ElderLife AI — الجزء الثالث: الرؤية الحاسوبية وكشف السقوط</title>
<style>
  :root{
    --ink:#16202a; --muted:#5d6b7a; --line:#e3e8ee; --bg:#f6f8fa;
    --blue:#1f6feb; --green:#1a7f37; --red:#b42318; --amber:#9a6700;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:"Segoe UI",Tahoma,Arial,sans-serif;line-height:1.75}
  .wrap{max-width:1080px;margin:0 auto;padding:32px 22px 80px}
  header.hero{background:linear-gradient(135deg,#0d2a4a,#1f6feb);color:#fff;
              border-radius:18px;padding:34px 30px;margin-bottom:26px}
  header.hero h1{margin:0 0 6px;font-size:29px;letter-spacing:-.3px}
  header.hero p{margin:0;opacity:.9;font-size:15px}
  .badges{margin-top:16px;display:flex;gap:8px;flex-wrap:wrap}
  .badge{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.25);
         padding:4px 11px;border-radius:999px;font-size:12.5px}
  h2{font-size:22px;margin:42px 0 14px;padding-bottom:8px;border-bottom:2px solid var(--line)}
  h3{font-size:17px;margin:24px 0 10px}
  p.note{color:var(--muted);font-size:14.5px}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
  .card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px 14px;text-align:center}
  .card .value{font-size:25px;font-weight:700;letter-spacing:-.5px}
  .card .label{font-size:12.5px;color:var(--muted);margin-top:2px}
  .card.good .value{color:var(--green)}
  .card.bad .value{color:var(--red)}
  .card.warn .value{color:var(--amber)}
  table{width:100%;border-collapse:collapse;background:#fff;border-radius:12px;
        overflow:hidden;font-size:14.5px;margin:12px 0}
  th,td{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line)}
  th{background:#eef3f8;font-weight:600;font-size:13px}
  tr.hl td{background:#eaf6ee;font-weight:700}
  tr:last-child td{border-bottom:none}
  figure{margin:18px 0;background:#fff;border:1px solid var(--line);
         border-radius:14px;padding:12px}
  figure img{display:block;border-radius:8px}
  figcaption{color:var(--muted);font-size:13px;margin-top:8px;text-align:center}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:760px){.grid2{grid-template-columns:1fr}}
  pre{background:#0f1720;color:#d7e3ef;padding:16px;border-radius:12px;
      overflow-x:auto;font-size:13px;direction:ltr;text-align:left;line-height:1.55}
  code{background:#eef2f6;padding:1px 5px;border-radius:5px;font-size:13px;direction:ltr;display:inline-block}
  .callout{border-right:4px solid var(--blue);background:#eef5ff;padding:13px 16px;
           border-radius:10px;margin:16px 0;font-size:14.5px}
  .callout.warn{border-color:#d4a72c;background:#fff8e6}
  .callout.bad{border-color:var(--red);background:#fdeceb}
  .callout.good{border-color:var(--green);background:#eaf6ee}
  ol,ul{padding-right:22px}
  li{margin:6px 0}
  .pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600}
  .pill.ok{background:#e6f4ea;color:var(--green)}
  .pill.no{background:#fdeceb;color:var(--red)}
  .pill.mid{background:#fff4d6;color:var(--amber)}
  footer{margin-top:50px;color:var(--muted);font-size:13px;text-align:center;
         border-top:1px solid var(--line);padding-top:18px}
  .tree{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px;
        font-family:Consolas,monospace;font-size:13px;direction:ltr;text-align:left;white-space:pre}
</style>
</head>
<body>
<div class="wrap">
<header class="hero">
  <h1>ElderLife AI — الجزء الثالث</h1>
  <p>الرؤية الحاسوبية وكشف السقوط · من الداتا للـ API جاهز للربط</p>
  <div class="badges">
    <span class="badge">URFD · 70 حدث</span>
    <span class="badge">OpenCV + MediaPipe + PyTorch</span>
    <span class="badge">F1 = 0.951</span>
    <span class="badge">44 اختبار</span>
    <span class="badge">FastAPI + Docker</span>
  </div>
</header>
"""

FOOT = """
<footer>
  المشروع: ElderLife AI — الجزء الثالث (Computer Vision &amp; Fall Detection) ·
  كل الأرقام في هذه الصفحة held-out من تقييم 5-fold على مستوى الحدث.
</footer>
</div></body></html>
"""


def build() -> str:
    parts = [HEAD]

    # ---------------------------------------------------------------- overview
    parts.append("<h2>1. الخلاصة في 6 أرقام</h2>")
    parts.append(
        "<div class='cards'>"
        + card("0.951", "F1 — الموديل النهائي", "good")
        + card("0.967", "Recall — السقوط اللي اتلقط", "good")
        + card("0.943", "Precision — دقة الإنذار", "good")
        + card("0.992", "AUC (متوسط folds)")
        + card("70", "حدث: 30 سقوط + 40 نشاط")
        + card("24,641", "معامل — موديل صغير عن قصد")
        + "</div>"
    )
    parts.append(
        "<div class='callout good'><b>ملخص:</b> الموديل بيشوف فيديو من كاميرا ثابتة، "
        "بيستخرج حركة الشخص (OpenCV) ووضع جسمه (MediaPipe)، وبعدين شبكة زمنية صغيرة "
        "(TCN) بتقرر: فيه سقوط ولا لأ. القرار بيرجع JSON للباك إند مع سكور وتوقيت وسبب."
        "</div>"
    )

    # ---------------------------------------------------------------- scope
    parts.append("<h2>2. بنود التكليف — بدليل</h2>")
    parts.append(
        table(
            ["البند", "إزاي اتحقق", "الحالة"],
            [
                ["تجهيز داتا السقوط والنشاط اليومي", "URFD: 70 حدث، audit كامل، license موثّق",
                 "<span class='pill ok'>تم</span>"],
                ["Baseline يفرّق السقوط عن ADL", "rule + logistic regression + random forest",
                 "<span class='pill ok'>تم</span>"],
                ["OpenCV + PyTorch + pose estimation", "motion.py + pose.py + temporal.py",
                 "<span class='pill ok'>تم</span>"],
                ["تقييم Precision / Recall / F1", "5-fold على مستوى الحدث، thresholds جوه الـ fold",
                 "<span class='pill ok'>تم</span>"],
                ["Inference جاهز للنظام", "REST API + JSON contract + Docker + alert state machine",
                 "<span class='pill ok'>تم</span>"],
                ["تصنيف أنشطة متعدد الفئات (multi-class)", "غير مطلوب في التنفيذ الحالي — ADL مستخدمة كـ negatives",
                 "<span class='pill mid'>خارج النطاق</span>"],
            ],
        )
    )

    # ---------------------------------------------------------------- data
    parts.append("<h2>3. الداتا — وأول مشكلة اكتشفتها</h2>")
    parts.append(
        "<p class='note'>URFD فيها 100 ملف، لكن السقوط متصوّر بكاميرتين. لو قسّمت على "
        "الملفات، نفس السقوط هيبقى في التدريب والاختبار مع بعض — <b>تسريب كامل</b>. "
        "عشان كده التقييم على cam0 بس = 70 حدث.</p>"
    )
    parts.append(
        "<div class='cards'>"
        + card("100", "ملف فيديو")
        + card("70", "حدث حقيقي")
        + card("8.3", "دقيقة إجمالًا")
        + card("640×240", "depth + RGB جنب بعض")
        + "</div>"
    )
    parts.append(
        "<div class='callout warn'>الفريم الأصلي <b>640×240</b> = عمق على الشمال + RGB على "
        "اليمين. لو استخدمت الفريم كامل، الموديل هيشوف صورة العمق مع الصورة العادية. "
        "القص بيحصل مرة واحدة في <code>src/data/video.py</code>.</div>"
    )

    # ---------------------------------------------------------------- bugs
    parts.append("<h2>4. مشكلتان كانتا هتخلي النتائج كذّابة</h2>")
    parts.append(
        "<h3>أ) تلوّث حالة التتبع بين المقاطع</h3>"
        "<p class='note'>MediaPipe في وضع الفيديو بيحتفظ بحالة تتبع من المقطع السابق. "
        "النتيجة: هيكل عظمي متجمّد ومستحيل تشريحيًا (الورك تحت الكعب) — ومع ذلك "
        "<code>detected = 1</code>. داتا فاسدة <b>تبدو سليمة</b>.</p>"
    )
    parts.append(
        img("pose_check", "لقطة كشفت المشكلة: النقاط على الشخص فعلًا، لكن الأرقام في الـ CSV "
                          "كانت مضغوطة في آخر 25% من الصورة — فحص بصري مقابل الأرقام.", "82%")
    )
    parts.append(
        "<div class='callout bad'><b>الحل:</b> تشغيل MediaPipe في <code>IMAGE</code> mode "
        "(بدون حالة أصلًا) بدل VIDEO mode + تصغير الفريمات لـ 15fps. "
        "النتيجة: 70 مقطع في 7 دقايق بدل 30+ دقيقة، وداتا سليمة.</div>"
    )
    parts.append(
        "<h3>ب) shortcut طول المقطع</h3>"
        "<p class='note'>أول نسخة من الموديل الزمني طلعت <b>F1 = 0.966 و AUC = 1.000 في كل "
        "fold</b>. ده مريب. الفحص كشف إن سقوط URFD متوسطه 50 فريم والنشاط اليومي 112 — "
        "فالموديل كان بيتعلم <b>طول الفيديو</b> مش السلوك.</p>"
    )
    parts.append(
        table(
            ["التجربة", "F1"],
            [
                ["طول المقطع لوحده (control)", "0.820"],
                ["الموديل على المقطع كامل", "0.966 ← ملوّث بالـ shortcut"],
                ["الموديل على windows ثابتة 3.2s", "<b>0.951 ← الرقم الأمين</b>"],
            ],
            highlight=2,
        )
    )
    parts.append(
        "<div class='callout'><b>الدرس:</b> الرقم الأعلى مش دايمًا الصح. الفرق بين 0.966 "
        "و 0.951 هو الفرق بين موديل هيقع على كاميرا حقيقية وموديل بيفهم.</div>"
    )

    # ---------------------------------------------------------------- baseline
    parts.append("<h2>5. المرحلة 1 — الـ baselines</h2>")
    parts.append(
        table(
            ["الموديل", "Precision", "Recall", "F1"],
            [
                ["Rule state machine (متظبّط جوه كل fold)", "0.827", "0.733", "0.768"],
                ["Logistic Regression (motion + pose)", "0.950", "0.933", "0.935"],
                ["Random Forest (motion + pose)", "0.888", "0.833", "0.838"],
            ],
            highlight=1,
        )
    )
    parts.append(
        "<h3>مين اللي بيجيب النتيجة؟ (ablation)</h3>"
        + table(
            ["الميزات", "logreg F1", "rf F1"],
            [
                ["Motion بس (OpenCV)", "0.858", "0.791"],
                ["Pose بس (MediaPipe)", "0.861", "0.833"],
                ["الاتنين مع بعض", "<b>0.935</b>", "0.838"],
            ],
            highlight=2,
        )
        + "<p class='note'>الاتنين مكمّلين لبعض — مش تكرار. دي الحجة إن الـ pose estimation "
          "مضاف بقيمة حقيقية.</p>"
    )
    parts.append(
        "<div class='grid2'>"
        + img("cm_logreg", "Confusion matrix — Logistic Regression")
        + img("cm_rule", "Confusion matrix — Rule baseline")
        + "</div>"
    )

    # ---------------------------------------------------------------- temporal
    parts.append("<h2>6. المرحلة 2 — الموديل الزمني (المستخدم)</h2>")
    parts.append(
        table(
            ["الموديل", "Precision", "Recall", "F1", "AUC"],
            [
                ["<b>Temporal TCN</b>", "0.943", "0.967", "<b>0.951</b>", "0.992"],
                ["LogReg (summary)", "0.933", "0.933", "0.933", "—"],
                ["Random Forest", "0.862", "0.833", "0.847", "—"],
                ["Rule", "0.846", "0.733", "0.786", "—"],
            ],
            highlight=0,
        )
    )
    parts.append(
        "<div class='cards'>"
        + card("19", "قناة لكل فريم")
        + card("3.2s", "طول الـ window")
        + card("61", "فريم = مجال الرؤية (≈4.1s)")
        + card("24,641", "معامل")
        + "</div>"
    )
    parts.append(
        "<p class='note'>الفكرة: السقوط مش صورة — السقوط <b>تسلسل</b>: واقف → هبوط سريع → "
        "يبقى على الأرض. الشبكة بتاخد 19 قناة لكل فريم (وضع الجسم + حركة الـ blob) وبتجمّع "
        "القرار على مدى 4 ثواني.</p>"
    )
    parts.append(img("cm_temporal", "التوزيع النهائي: 29 سقوط اتلقط، 2 إنذار كاذب، سقوط واحد فايت"))

    # ---------------------------------------------------------------- errors
    parts.append("<h2>7. الأخطاء — وسببها الحقيقي</h2>")
    parts.append(
        "<p class='note'>كل حالة غلط راجعتها بصريًا، مش تخمين.</p>"
    )
    parts.append(
        "<h3>fall-21 و fall-23: سقوط ناحية الكاميرا</h3>"
        "<p class='note'>الجسم بيتقصّر في الصورة (foreshortening)، فالجذع مايبقاش أفقي أبدًا "
        "(الزاوية وصلت 26-31° بس). كاميرا واحدة أمامية <b>ماديًا</b> بتفقد الإشارة.</p>"
    )
    parts.append(
        "<div class='grid2'>"
        + img("fall23", "fall-23: السقوط بيحصل ناحية الكاميرا — الشكل النهائي مش أفقي")
        + img("tl_fall21", "الموديل الزمني: إشارة واضحة رغم إن الوضع النهائي مش أفقي")
        + "</div>"
    )
    parts.append(
        "<div class='callout good'><b>تحسّن حقيقي:</b> fall-21 و fall-23 كانوا فايتين في "
        "الـ baseline والـ rule. الموديل الزمني صطادهم، لأن الهبوط باين كتسلسل حتى لو الوضع "
        "النهائي متشوّه.</div>"
    )
    parts.append(
        "<h3>adl-30 و adl-33: نشاط يومي بيوصل الأرض</h3>"
        "<p class='note'>هندسيًا مطابق للسقوط في لحظة الهبوط. اللي يفرّقهم هو <b>السياق</b>: "
        "السقوط مسبوقش بجلوس متحكَّم فيه، ومش بيتبعه قومة.</p>"
    )
    parts.append(
        "<div class='grid2'>"
        + img("adl33", "adl-33: الشخص بينزل الأرض باختياره")
        + img("tl_adl30", "adl-30: إشارة موجودة لكن أقل من حد الإنذار (0.17)")
        + "</div>"
    )
    parts.append(
        "<div class='callout warn'><b>الخلاصة:</b> 3 أخطاء من 70، وكلهم قابلين للتفسير. "
        "مفيش خطأ غامض — ودي علامة إن النظام مفهوم مش صندوق أسود.</div>"
    )

    # ---------------------------------------------------------------- inference
    parts.append("<h2>8. المرحلة 3 — الـ API</h2>")
    parts.append(
        "<pre>POST /analyze/urfd   (multipart: file)\n\n"
        "{\n"
        '  "schema_version": "1.0",\n'
        '  "is_fall": true,\n'
        '  "score": 0.999,\n'
        '  "threshold": 0.5,\n'
        '  "confidence": "high",\n'
        '  "onset_s": 1.6,\n'
        '  "windows": [{"start_s": 1.6, "end_s": 4.8, "score": 0.999}],\n'
        '  "pose_coverage": 0.75,\n'
        '  "latency_ms": 996.0,\n'
        '  "model": {"name": "elderlife-fall-tcn", "version": "1.0.0"}\n'
        "}</pre>"
    )
    parts.append(
        "<div class='callout good'><b>متجرّب فعليًا:</b> الخدمة اشتغلت بـ uvicorn وردّت على "
        "طلب HTTP حقيقي: <code>is_fall=true, score=0.999, latency=996ms</code>.</div>"
    )
    parts.append(
        "<div class='grid2'>"
        + img("inf_fall", "سقوط: إشارة عالية عبر الـ windows")
        + img("inf_adl", "نشاط عادي: إشارة منخفضة (0.17) — مفيش إنذار")
        + "</div>"
    )
    parts.append(
        "<h3>state machine للإنذار</h3>"
        "<p class='note'>الموديل بيقول \"الـ window دي فيها سقوط؟\" — النظام لازم يقول "
        "\"أصحّي حد دلوقتي؟\". الطبقة دي بتعمل debounce + cooldown 30 ثانية، عشان السقوط "
        "الواحد يطلّع <b>إنذار واحد</b> مش إنذار كل 3 ثواني.</p>"
    )

    # ---------------------------------------------------------------- quality
    parts.append("<h2>9. جودة الكود</h2>")
    parts.append(
        "<div class='cards'>"
        + card(str(TEST_COUNT), "اختبار ناجح", "good")
        + card("4", "باجّات اكتشفتها الاختبارات", "warn")
        + card("0", "NaN مسموح في الميزات", "good")
        + "</div>"
    )
    parts.append(
        "<p class='note'>الاختبارات كشفت 4 مشاكل حقيقية قبل التسليم: تسريب الـ padding "
        "لقرار الموديل، حساب <code>frame_scores</code> غلط، تجاهل الـ mask في التطبيع، "
        "وسببين غير واضحين في state machine.</p>"
    )
    parts.append(
        "<div class='tree'>src/\n"
        "  data/        video reading, dataset audit\n"
        "  features/    pose.py (MediaPipe), motion.py (OpenCV)\n"
        "  models/      rule_baseline.py, sequences.py, temporal.py (TCN)\n"
        "  eval/        dataset.py, metrics.py\n"
        "  inference/   pipeline.py, service.py (FastAPI), alerting.py\n"
        "scripts/       extract_*, run_baseline, ablate, train_temporal, demo, build_showcase\n"
        "tests/         5 files, 44 tests\n"
        "models/        fall_tcn.pt (0.1 MB), pose_landmarker_lite.task (5.5 MB)\n"
        "reports/       6 reports + figures\n"
        "docs/          model-card, api-contract, pose-pipeline, evaluation-protocol\n"
        "Dockerfile · docker-compose.yml · requirements.txt</div>"
    )

    # ---------------------------------------------------------------- limits
    parts.append("<h2>10. القيود — قبل أي ادعاء</h2>")
    parts.append(
        "<ol>"
        "<li><b>70 حدث، ممثل واحد، غرفة واحدة، 8 دقايق.</b> مفيش تنوع ديموغرافي أو بيئي.</li>"
        "<li><b>سقوط ناحية الكاميرا</b> متشوّه — محتاج كاميرا تانية أو 3D lifting.</li>"
        "<li><b>الأنشطة الأرضية</b> (النوم/الجلوس على الأرض) هي مصدر الإنذارات الكاذبة.</li>"
        "<li><b>شخص واحد بس</b> في الكادر.</li>"
        "<li><b>كاميرا ثابتة</b> — تقدير الخلفية بعمل median للفيديو.</li>"
        "<li><b>مش real-time</b>: 1.5–3× أسرع من الريال تايم على CPU، مش 30fps live.</li>"
        "<li><b>مش جهاز طبي</b> — مكوّن فحص مساعد، والقرار النهائي لبشري.</li>"
        "</ol>"
    )
    parts.append(
        "<div class='callout bad'><b>الترخيص:</b> داتا URFD هي CC BY-NC-SA 4.0 — "
        "استخدام أكاديمي غير تجاري بس، والترخيص بيسري على الأوزان المدرّبة.</div>"
    )

    parts.append(FOOT)
    return "".join(parts)


def main() -> None:
    html_text = build()
    OUT.write_text(html_text, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
