# تسليم الجزء الثالث — Computer Vision & Fall Detection

> **النطاق:** الجزء ده بس. الباك إند / الفرونت إند / الداتابيز / Power BI **مش** جزء منه.
> الحد الفاصل بينك وبين باقي الفريق هو **JSON واحد** بيرجع من الـ API.

---

## 1. إيه اللي اتعمل

موديل كشف سقوط من فيديو كاميرا ثابتة، بيشتغل على:

```
فيديو → OpenCV (حركة الشخص) + MediaPipe (وضع الجسم) → شبكة زمنية TCN → قرار
```

| | |
|---|---|
| الداتا | URFD — 70 حدث (30 سقوط + 40 نشاط يومي)، كاميرا واحدة |
| الـ baseline | rule + logistic regression + random forest |
| الموديل النهائي | TCN، 24,641 معامل، نافذة 3.2 ثانية |
| **F1** | **0.951** (held-out، 5-fold على مستوى الحدث) |
| Recall | 0.967 — سقوط واحد فايت من 30 |
| Precision | 0.943 |

كل الأرقام في `reports/temporal_report.md` و `reports/baseline_report.md`.
عرض كامل في صفحة واحدة: `reports/showcase.html` (افتحها في المتصفح).

---

## 2. إيه اللي في الحزمة

```
src/
  data/          قراءة الفيديو + audit الداتا
  features/      pose.py (MediaPipe), motion.py (OpenCV)
  models/        rule_baseline.py, sequences.py, temporal.py (TCN)
  eval/          dataset.py, metrics.py
  inference/     pipeline.py, service.py (FastAPI), alerting.py
scripts/         استخراج الميزات، التدريب، التقييم، الديمو
tests/           44 اختبار
models/          fall_tcn.pt (الأوزان), pose_landmarker_lite.task
reports/         التقارير + الصور + showcase.html
docs/            model-card, api-contract, pose-pipeline, evaluation-protocol
Dockerfile, docker-compose.yml, requirements.txt
```

**مش موجود في الحزمة (عن قصد):** `data/` — فيديوهات URFD 115 ميجا وترخيصها
CC BY-NC-SA 4.0 (أكاديمي غير تجاري). لازم تنزّلها بنفسك بالأمر اللي تحت.

---

## 3. التشغيل من الصفر

```powershell
# 1. المتطلبات: Python 3.11 + ffmpeg
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # بياخد وقت (torch ~200MB)

# 2. الداتا (115MB، السيرفر بطيء)
powershell -ExecutionPolicy Bypass -File scripts/download_urfd.ps1
python -m src.data.audit_dataset

# 3. الميزات (motion ~40 ثانية | pose ~7 دقايق)
python -m scripts.extract_motion_features
python -m scripts.extract_pose_features --camera 0 --upscale 2 --stride 2

# 4. التأكد إن كل حاجة سليمة
python -m pytest tests -q                # المفروض: 44 passed

# 5. ديمو على فيديوهات حقيقية
python -m scripts.demo_inference
```

**لو مش عايز تعيد التدريب:** الأوزان جاهزة في `models/fall_tcn.pt`، تقدر تشغّل
الخطوة 5 و6 على طول بعد الداتا.

---

## 4. تشغيل الـ API

```powershell
uvicorn src.inference.service:app --host 0.0.0.0 --port 8100
# أو
docker compose up --build
```

```bash
curl -X POST "http://localhost:8100/analyze?layout=urfd" -F "file=@clip.mp4"
```

```json
{
  "schema_version": "1.0",
  "is_fall": true,
  "score": 0.999,
  "threshold": 0.5,
  "confidence": "high",
  "onset_s": 1.6,
  "windows": [{"start_s": 1.6, "end_s": 4.8, "score": 0.999}],
  "pose_coverage": 0.75,
  "latency_ms": 996.0,
  "model": {"name": "elderlife-fall-tcn", "version": "1.0.0"}
}
```

التفاصيل الكاملة + أكواد الأخطاء: `docs/api-contract.md`.

---

## 5. المطلوب منك في الباك إند (مهم)

1. **Tت5لي tracker لكل كاميرا.** `src/inference/alerting.py::FallEventTracker` فيه
   debounce + cooldown. الحالة بتاعته per-stream — كاميرتين مش يشاركوا نفس الـ tracker.
2. **متبعتش إنذار على `is_fall` لوحده.** الـ tracker موجود عشان يمنع الإنذار المتكرر
   لنفس الحادثة.
3. **خزّن `model.version` و `threshold` مع كل إنذار** — من غيرهم مش هتقدر تفسّر إنذار
   قديم بعد تحديث الموديل.
4. **الـ API بترجع سكور وتوقيت، مش صور.** privacy by design.
5. **اللانسي:** 1.5–5 ثانية للفيديو الواحد (مش real-time). للكاميرا الحية، قطع نوافذ
   3 ثواني وابعتهم وهي بتيجي.

---

## 6. حاجات لازم تعرفها قبل ما تسلّم

| الحاجة | الحالة |
|---|---|
| تصنيف أنشطة متعدد الفئات (multi-class) | ❌ مش معمول — الـ ADL مستخدمة كـ negatives بس |
| Docker | الملفات جاهزة، بس **متبنتش** (Docker مش متسطب على جهاز التطوير) |
| ربط فعلي بالباك إند | ❌ لسه — الـ contract بس |
| Streaming adapter لكاميرا live | ❌ لسه — `analyze_frames()` موجودة، محتاجة wrapper |
| real-time 30fps | ❌ 1.5–3× أسرع من الريال تايم، مش live |

اقرا `docs/model-card.md` قسم **Limitations** قبل أي عرض للدكتور.

---

## 7. إعادة التدريب / التقييم

```powershell
python -m scripts.run_baseline       # rule + logreg + RF
python -m scripts.ablate_features    # مين اللي بيجيب النتيجة
python -m scripts.train_temporal     # الموديل النهائي (~4 دقايق CPU)
python -m scripts.error_analysis     # الأخطاء بالاسم
python -m scripts.build_showcase     # يبني صفحة العرض
```

---

## 8. الترخيص

داتا **URFD** مرخّصة **CC BY-NC-SA 4.0** — أكاديمي غير تجاري. الترخيص بيسري على
الأوزان المدرّبة كمان. الاستخدام التجاري محتاج داتا تانية.

الاقتباس:
> Kwolek, B., Kepski, M. (2014). *Human fall detection on embedded platform using depth
> maps and wireless accelerometer.* Computer Methods and Programs in Biomedicine, 117(3),
> 489–501.
