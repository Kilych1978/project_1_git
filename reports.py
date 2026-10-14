# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC — MODULAR COMPONENT
# MODULE: reports.py
# ============================================================
# Russian report copies, sync-effectiveness reports, and the pre-apply user confirmation.
#
# This module is part of the modular version of the project.
# Do not run it directly; start the pipeline with start_sync.py.
# ============================================================

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from config import *
from common import *

def _translate_report_line_to_russian(line: str) -> str:
    """Переводит человекочитаемые строки отчётов, сохраняя технические идентификаторы и пути."""
    leading = line[:len(line) - len(line.lstrip())]
    stripped = line.strip()
    if not stripped:
        return line

    # Пути и имена файлов не переводим. Это особенно важно для Windows-путей,
    # в которых встречаются английские слова и названия кодеков.
    path_like = (
        (len(stripped) >= 3 and stripped[1:3] == ':\\')
        or stripped.startswith('C:\\')
        or stripped.startswith('\\\\')
    )

    exact = {
        "FINAL AUDIO RENDER": "ФИНАЛЬНЫЙ РЕНДЕР АУДИО",
        "FINAL RENDERED AUDIO VALIDATION": "ФИНАЛЬНАЯ ПРОВЕРКА ГОТОВОЙ АУДИОДОРОЖКИ",
        "INDEPENDENT AUDIO SYNC VALIDATION": "НЕЗАВИСИМАЯ ПРОВЕРКА СИНХРОНИЗАЦИИ АУДИО",
        "FINAL ROBUST SYNCHRONIZATION MAP": "ФИНАЛЬНАЯ УСТОЙЧИВАЯ КАРТА СИНХРОНИЗАЦИИ",
        "UNIVERSAL AUDIO WAVEFORM SYNC": "УНИВЕРСАЛЬНАЯ СИНХРОНИЗАЦИЯ АУДИО ПО ФОРМЕ ВОЛНЫ",
        "CONFIDENCE-AWARE SYNCHRONIZATION": "СИНХРОНИЗАЦИЯ С УЧЁТОМ НАДЁЖНОСТИ ИЗМЕРЕНИЙ",
        "PRE-APPLY SYNCHRONIZATION REPORT": "ОТЧЁТ О СИНХРОНИЗАЦИИ ПЕРЕД ПРИМЕНЕНИЕМ",
        "MANUAL AUDACITY CHECK": "РУЧНАЯ ПРОВЕРКА В AUDACITY",
        "NON-REGRESSION GUARD REPORT": "ОТЧЁТ О ЗАЩИТЕ ОТ УХУДШЕНИЯ",
        "SYNC EFFECTIVENESS REPORT": "ОТЧЁТ ОБ ЭФФЕКТИВНОСТИ СИНХРОНИЗАЦИИ",
        "FINAL SYNC EFFECTIVENESS REPORT": "ФИНАЛЬНЫЙ ОТЧЁТ ОБ ЭФФЕКТИВНОСТИ СИНХРОНИЗАЦИИ",
        "RENDER-AWARE NON-REGRESSION REPORT": "ОТЧЁТ О ЗАЩИТЕ ОТ УХУДШЕНИЯ ПО РЕАЛЬНОМУ РЕНДЕРУ",
        "RENDER-AWARE NON-REGRESSION": "ЗАЩИТА ОТ УХУДШЕНИЯ ПО РЕАЛЬНОМУ РЕНДЕРУ",
        "PRESERVE_TARGET": "СОХРАНЕНИЕ ИСХОДНОЙ ДОРОЖКИ",
        "FALLBACK_BASELINE": "ВОЗВРАТ К ИСХОДНОЙ ДОРОЖКЕ",
        "NO_CHANGE": "БЕЗ ИЗМЕНЕНИЙ",
        "MIXED_SAFE_MAP": "СМЕШАННАЯ БЕЗОПАСНАЯ КАРТА",
        "MIXED_SAFE_MAP_DENSE": "СМЕШАННАЯ БЕЗОПАСНАЯ КАРТА НА ПЛОТНОЙ СЕТКЕ",
        "LOCAL_QUARANTINE_SAFE_MAP": "БЕЗОПАСНАЯ КАРТА С ЛОКАЛЬНЫМ КАРАНТИНОМ",
        "FULL_CANDIDATE": "ПОЛНАЯ КАРТА-КАНДИДАТ",
        "GOOD": "ХОРОШО",
        "EXCELLENT": "ОТЛИЧНО",
        "ACCEPTABLE": "ПРИЕМЛЕМО",
        "BAD": "ПЛОХО",
        "none": "нет",
        "True": "ДА",
        "False": "НЕТ",
    }
    if stripped in exact:
        return leading + exact[stripped]

    if path_like:
        # В строках вида "OUTPUT: C:\..." переводим только подпись, не путь.
        return line

    text = stripped

    # Защищаем имена файлов и пути в любой позиции строки. Иначе общий перевод
    # слов вроде REPORT / VERSION / DELAY может испортить имя артефакта.
    protected_tokens = []

    def _protect_token(match):
        token = match.group(0)
        placeholder = f"__REPORT_TOKEN_{len(protected_tokens)}__"
        protected_tokens.append(token)
        return placeholder

    # Полные Windows-пути.
    text = re.sub(r"[A-Za-z]:\\[^\r\n]+", _protect_token, text)
    # Имена файлов с расширениями, включая отчёты и графики.
    text = re.sub(
        r"(?<![A-Za-z0-9_])[^\s:]+\.(?:txt|jpg|jpeg|png|csv|json|wav|dts|ac3|flac|pcm)(?![A-Za-z0-9_])",
        _protect_token,
        text,
        flags=re.IGNORECASE,
    )

    def _restore_tokens(value: str) -> str:
        for index, token in enumerate(protected_tokens):
            value = value.replace(f"__REPORT_TOKEN_{index}__", token)
        return value

    # Сначала переводим длинные фразы, пока отдельные слова ещё не были заменены.
    early_phrases = [
        (
            "КАРТА DIFFERENCE FROM PREVIOUS VERSION:",
            "РАЗНИЦА С ПРЕДЫДУЩЕЙ ВЕРСИЕЙ КАРТЫ:",
        ),
        (
            "ТОЛКОВАНИЕ ДЛИТЕЛЬНОСТИ: ДЛИТЕЛЬНОСТЬ ЗАКОДИРОВАННОГО ВЫХОДА is reported separately from synchronization residuals.",
            "ТОЛКОВАНИЕ ДЛИТЕЛЬНОСТИ: длительность закодированного выхода указана отдельно от остаточной ошибки синхронизации.",
        ),
        (
            "Подтверждённых расхождений means both validation sides independently support a deviation from the КАРТА.",
            "Подтверждённые расхождения означают, что обе стороны проверки независимо подтверждают отклонение от КАРТЫ.",
        ),
        (
            "Неоднозначных CANDIDATE means the waveform search did not provide enough independent evidence to confirm a КАРТА failure.",
            "Неоднозначный КАНДИДАТ означает, что поиск по форме волны не дал достаточно независимых данных для подтверждения ошибки КАРТЫ.",
        ),
        (
            "Неоднозначных candidates do not modify the synchronization КАРТА and are diagnostic only.",
            "Неоднозначные кандидаты не изменяют КАРТУ СИНХРОНИЗАЦИИ и используются только для диагностики.",
        ),
        (
            "The guard compares candidate and SAFE КАРТА on the SAME dense independent validation grid. Rejected ТОЧЕК are hard barriers; correction cannot interpolate through them.",
            "Защита сравнивает карту-кандидат и БЕЗОПАСНУЮ КАРТУ на одной и той же плотной независимой сетке проверки. Отклонённые точки являются жёсткими барьерами: коррекция не может интерполироваться через них.",
        ),
        (
            "The guard compares candidate and SAFE MAP on the SAME dense independent validation grid. Rejected points are hard barriers; correction cannot interpolate through them.",
            "Защита сравнивает карту-кандидат и БЕЗОПАСНУЮ КАРТУ на одной и той же плотной независимой сетке проверки. Отклонённые точки являются жёсткими барьерами: коррекция не может интерполироваться через них.",
        ),
        (
            "DURATION INTERPRETATION: encoded output duration is reported separately from synchronization residuals.",
            "ТОЛКОВАНИЕ ДЛИТЕЛЬНОСТИ: длительность закодированного выхода указана отдельно от остаточной ошибки синхронизации.",
        ),
        (
            "FORMAT DECISION: Exact copy; no re-encoding because the non-regression guard found no safe timing correction.",
            "РЕШЕНИЕ ПО ФОРМАТУ: точная копия; перекодирование не выполнялось, поскольку защита от ухудшения не обнаружила безопасной временной коррекции.",
        ),
        (
            "ROBUST ВЕРДИКТ КАРТЫ:",
            "УСТОЙЧИВЫЙ ВЕРДИКТ КАРТЫ:",
        ),
        (
            "APPLY КАРТА:",
            "КАРТА ПРИМЕНЕНИЯ:",
        ),
        (
            "HARD candidates: 0",
            "ЖЁСТКИХ КАНДИДАТОВ: 0",
        ),
        (
            "HARD OUTLIERS:",
            "ЖЁСТКИЕ ВЫБРОСЫ:",
        ),
        (
            "ROBUST VERDICT: ОТЛИЧНО",
            "УСТОЙЧИВЫЙ ВЕРДИКТ: ОТЛИЧНО",
        ),
        (
            "Подтверждённых расхождений means both validation sides independently support a deviation from the MAP.",
            "Подтверждённые расхождения означают, что обе стороны проверки независимо подтверждают отклонение от КАРТЫ.",
        ),
        (
            "Неоднозначных CANDIDATE means the waveform search did not provide enough independent evidence to confirm a MAP failure.",
            "Неоднозначный КАНДИДАТ означает, что поиск по форме волны не дал достаточно независимых данных для подтверждения ошибки КАРТЫ.",
        ),
        (
            "Неоднозначных candidates do not modify the synchronization MAP and are diagnostic only.",
            "Неоднозначные кандидаты не изменяют КАРТУ СИНХРОНИЗАЦИИ и используются только для диагностики.",
        ),
        (
            "The guard compares candidate and SAFE MAP on the SAME dense independent validation grid. Rejected points are hard barriers; correction cannot interpolate through them.",
            "Защита сравнивает карту-кандидат и БЕЗОПАСНУЮ КАРТУ на одной и той же плотной независимой сетке проверки. Отклонённые точки являются жёсткими барьерами: коррекция не может интерполироваться через них.",
        ),
    ]
    for old, new in early_phrases:
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)

    # Полные фразы. Регулярное сравнение без учёта регистра закрывает старые
    # варианты написания в отчётах разных версий.
    phrases = [
        ("Baseline measurement: symmetric bidirectional validation", "Измерение исходной синхронизации: симметричная двусторонняя проверка"),
        ("Weak and invalid waveform measurements are NOT allowed to move the main map.", "Слабые и недействительные измерения формы волны НЕ могут изменять основную карту."),
        ("The source TARGET file remains untouched.", "Исходный файл TARGET остался без изменений."),
        ("The source TARGET remains untouched.", "Исходный файл TARGET остался без изменений."),
        ("The source TARGET has NOT been modified.", "Исходный TARGET НЕ изменялся."),
        ("SOURCE TARGET has NOT been modified.", "Исходный TARGET НЕ изменялся."),
        ("Minimum meaningful correction for already-synchronized TARGET: 0.006000 s", "Минимальная значимая коррекция для уже синхронной целевой дорожки: 0.006000 с"),
        ("Minimum meaningful improvement for already-synchronized TARGET: 0.004000 s", "Минимальное значимое улучшение для уже синхронной целевой дорожки: 0.004000 с"),
        ("Local quarantine may exclude a small problematic region without cancelling unrelated confirmed improvements.", "Локальный карантин может исключить небольшой проблемный участок, не отменяя подтверждённые улучшения в других местах."),
        ("The original TARGET has NOT been modified.", "Исходный TARGET НЕ изменялся."),
        ("The original TARGET was NOT modified.", "Исходный TARGET НЕ изменялся."),
        ("The original TARGET was not modified.", "Исходный TARGET НЕ изменялся."),
        ("The original TARGET remains untouched.", "Исходный TARGET остался без изменений."),
        ("Audio rendering has NOT been performed.", "Рендер аудио НЕ выполнялся."),
        ("Audio correction has NOT been applied yet.", "Коррекция аудио ещё НЕ применялась."),
        ("The next stage will create a new file.", "Следующий этап создаст новый файл."),
        ("For the rendered file the expected synchronization offset is 0 s.", "Для готового файла ожидаемое смещение синхронизации равно 0 с."),
        ("Exact copy; no re-encoding because the non-regression guard found no safe timing correction.", "Точная копия; перекодирование не выполнялось, поскольку защита от ухудшения не обнаружила безопасной временной коррекции."),
        ("Original codec family preserved.", "Семейство исходного кодека сохранено."),
        ("The threshold percentages describe the share of independent validation samples inside each residual band. They are not a literal percentage of perceived audio quality.", "Процентные значения показывают долю независимых контрольных измерений, попавших в соответствующий диапазон остаточной ошибки. Они не являются буквальным процентом воспринимаемого качества синхронизации."),
        ("The threshold percentages describe the share of independent validation samples inside each residual band; they are not a literal percentage of audible waveform similarity.", "Процентные значения показывают долю независимых контрольных измерений, попавших в соответствующий диапазон остаточной ошибки; это не буквальный процент слышимой схожести формы волны."),
        ("Percentages describe the independent validation samples that fall within the stated residual threshold; they are not a literal percentage of audible waveform similarity.", "Процентные значения показывают долю независимых контрольных измерений, попавших в указанный диапазон остаточной ошибки; это не буквальный процент слышимой схожести формы волны."),
        ("No HARD validation point was confirmed as a real map discrepancy by bidirectional validation.", "Ни одна контрольная точка класса HARD не была подтверждена двусторонней проверкой как реальное расхождение карты."),
        ("Candidate corrections are allowed into the final map only when dense independent validation demonstrates a clear improvement and enough neighboring validation points support the same correction. Local quarantine may exclude a small problematic region without cancelling unrelated confirmed improvements.", "Коррекция попадает в итоговую карту только тогда, когда плотная независимая проверка показывает явное улучшение и достаточное число соседних контрольных точек подтверждает ту же коррекцию. Локальный карантин может исключить небольшой проблемный участок, не отменяя подтверждённые улучшения в других местах."),
        ("SAFE MAP passes dense same-grid non-regression: no independent validation point was materially worsened.", "БЕЗОПАСНАЯ КАРТА прошла плотную проверку на одной сетке без ухудшения: ни одна независимая контрольная точка не была заметно ухудшена."),
        ("SAFE MAP passes outside local quarantine zones: all non-quarantined independent validation points are non-regressive and material improvements remain.", "БЕЗОПАСНАЯ КАРТА проходит проверку вне локальных зон карантина: ни одна независимая контрольная точка вне карантина не ухудшена, а заметные улучшения сохранены."),
        ("SAFE MAP passes outside local quarantine zones, but no material independent improvement remains outside those zones.", "БЕЗОПАСНАЯ КАРТА проходит проверку вне локальных зон карантина, но вне этих зон не осталось заметного независимого улучшения."),
        ("SAFE MAP still materially worsens at least one non-quarantined dense independent validation point.", "БЕЗОПАСНАЯ КАРТА по-прежнему заметно ухудшает как минимум одну плотную независимую контрольную точку вне зон карантина."),
        ("SAFE MAP would materially worsen at least one dense independent validation point.", "БЕЗОПАСНАЯ КАРТА заметно ухудшит как минимум одну плотную независимую контрольную точку."),
        ("Safe map passed non-regression check.", "Безопасная карта прошла проверку без ухудшения."),
        ("Safe map would materially worsen at least one independent validation point.", "Безопасная карта заметно ухудшит как минимум одну независимую контрольную точку."),
        ("no render", "рендер не выполнялся"),
        ("tiny corrections are suppressed on an already synchronized TARGET.", "мелкие коррекции подавляются для уже синхронной целевой дорожки."),
        ("The rendered file has NOT been modified.", "Готовый файл НЕ изменялся."),
        ("map-only placement", "размещение только по карте"),
        ("no re-encoding", "перекодирование не выполнялось"),
    ]
    for old, new in phrases:
        if re.search(re.escape(old) + r"$", text, flags=re.IGNORECASE):
            return leading + new
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)

    # Поля, которые почти всегда имеют значение после двоеточия. Значение справа
    # сохраняется без изменений: там могут быть имена файлов, кодеки и числа.
    labels = [
        ("REFERENCE:", "ЭТАЛОННАЯ ДОРОЖКА:"),
        ("TARGET:", "ЦЕЛЕВАЯ ДОРОЖКА:"),
        ("OUTPUT:", "ВЫХОДНОЙ ФАЙЛ:"),
        ("SOURCE TARGET:", "ИСХОДНАЯ ЦЕЛЕВАЯ ДОРОЖКА:"),
        ("SOURCE CODEC:", "ИСХОДНЫЙ КОДЕК:"),
        ("OUTPUT ENCODER:", "КОДЕК ВЫХОДА:"),
        ("OUTPUT CONTAINER:", "КОНТЕЙНЕР ВЫХОДА:"),
        ("PCM LOSSLESS FALLBACK:", "ЗАПАСНОЙ ВЫХОД PCM БЕЗ ПОТЕРЬ:"),
        ("FORMAT DECISION:", "РЕШЕНИЕ ПО ФОРМАТУ:"),
    ]
    for old, new in labels:
        if re.match(rf"^{re.escape(old)}", text, flags=re.IGNORECASE):
            tail = text[len(old):]
            return leading + _restore_tokens(new + tail)

    # Набор подписей и статусов, встречающихся в отчётах проекта.
    replacements = [
        ("ROBUST VERDICT", "УСТОЙЧИВЫЙ ВЕРДИКТ"),
        ("MAP VERDICT", "ВЕРДИКТ КАРТЫ"),
        ("MAP FIT", "КАЧЕСТВО ПОДГОНКИ КАРТЫ"),
        ("ANCHOR QUALITY", "КАЧЕСТВО ЯКОРЕЙ"),
        ("VALIDATION OUTLIER CANDIDATES", "КАНДИДАТОВ НА ВЫБРОСЫ ПРОВЕРКИ"),
        ("Candidate HARD outliers", "HARD-выбросов кандидата"),
        ("Confirmed discrepancies", "Подтверждённых расхождений"),
        ("CONFIRMED HARD DISCREPANCIES", "ПОДТВЕРЖДЁННЫХ HARD-РАСХОЖДЕНИЙ"),
        ("AMBIGUOUS HARD CANDIDATES", "НЕОДНОЗНАЧНЫХ HARD-КАНДИДАТОВ"),
        ("AMBIGUOUS VALIDATION POINTS", "НЕОДНОЗНАЧНЫХ ТОЧЕК ПРОВЕРКИ"),
        ("HARD candidates requiring manual review", "HARD-кандидатов, требующих ручной проверки"),
        ("Confirmed map", "Подтверждённых совпадений"),
        ("Confirmed discrepancy", "Подтверждённых расхождений"),
        ("Consistent but weak", "Согласованных, но слабых"),
        ("Ambiguous", "Неоднозначных"),
        ("Validation outlier candidates", "Кандидатов на выбросы проверки"),
        ("Max absolute difference", "Максимальная абсолютная разница"),
        ("Mean absolute difference", "Средняя абсолютная разница"),
        ("Median absolute difference", "Медианная абсолютная разница"),
        ("Maximum absolute map correction", "Максимальный модуль коррекции карты"),
        ("Start map offset", "Начальное смещение карты"),
        ("End map offset", "Конечное смещение карты"),
        ("Start offset", "Начальное смещение"),
        ("End offset", "Конечное смещение"),
        ("Total change", "Общее изменение"),
        ("Min offset", "Минимальное смещение"),
        ("Max offset", "Максимальное смещение"),
        ("Mean error", "Средняя ошибка"),
        ("Median error", "Медианная ошибка"),
        ("Max error", "Максимальная ошибка"),
        ("Mean residual", "Средний остаток"),
        ("Median residual", "Медианный остаток"),
        ("Max residual", "Максимальный остаток"),
        ("P90 residual", "P90 остатка"),
        ("P95 residual", "P95 остатка"),
        ("Mean correlation", "Средняя корреляция"),
        ("Median correlation", "Медианная корреляция"),
        ("Error improvement", "Улучшение ошибки"),
        ("Mean error improvement", "Улучшение средней ошибки"),
        ("Correlation gain", "Изменение корреляции"),
        ("Residual reduction", "Снижение остаточной ошибки"),
        ("Within 5 ms", "В пределах 5 мс"),
        ("Within 10 ms", "В пределах 10 мс"),
        ("Within 20 ms", "В пределах 20 мс"),
        ("Within 50 ms", "В пределах 50 мс"),
        ("Within 100 ms", "В пределах 100 мс"),
        ("Within 250 ms", "В пределах 250 мс"),
        ("AFTER mean residual", "После - средний остаток"),
        ("AFTER median residual", "После - медианный остаток"),
        ("AFTER max residual", "После - максимальный остаток"),
        ("AFTER mean correlation", "После - средняя корреляция"),
        ("AFTER within 50ms", "После, в пределах 50 мс"),
        ("BEFORE APPLY", "ПЕРЕД ПРИМЕНЕНИЕМ КАРТЫ"),
        ("AFTER APPLY", "ПОСЛЕ ПРИМЕНЕНИЯ КАРТЫ"),
        ("INDEPENDENT VALIDATION BEFORE APPLY", "НЕЗАВИСИМАЯ ПРОВЕРКА ПЕРЕД ПРИМЕНЕНИЕМ КАРТЫ"),
        ("INDEPENDENT VALIDATION AFTER MAP", "НЕЗАВИСИМАЯ ПРОВЕРКА ПОСЛЕ ПРИМЕНЕНИЯ КАРТЫ"),
        ("FINAL RENDER VALIDATION", "ФИНАЛЬНАЯ ПРОВЕРКА ГОТОВОГО РЕНДЕРА"),
        ("RENDER TIME", "ВРЕМЯ РЕНДЕРА"),
        ("COPY TIME", "ВРЕМЯ КОПИРОВАНИЯ"),
        ("REFERENCE TIMELINE DURATION", "ДЛИТЕЛЬНОСТЬ ШКАЛЫ ЭТАЛОННОЙ ДОРОЖКИ"),
        ("ENCODED OUTPUT DURATION", "ДЛИТЕЛЬНОСТЬ ЗАКОДИРОВАННОГО ВЫХОДА"),
        ("ENCODED DURATION DELTA", "РАЗНИЦА ДЛИТЕЛЬНОСТЕЙ ЗАКОДИРОВАННОГО ВЫХОДА И ЭТАЛОНА"),
        ("KNOWN CODEC FRAME DURATION", "ИЗВЕСТНАЯ ДЛИТЕЛЬНОСТЬ КАДРА КОДЕКА"),
        ("NEAREST CODEC FRAME MULTIPLE", "БЛИЖАЙШЕЕ КРАТНОЕ ДЛИТЕЛЬНОСТИ КАДРА КОДЕКА"),
        ("FRAME-GRANULARITY RESIDUAL", "ОСТАТОК ОКРУГЛЕНИЯ ДО ШАГА КАДРА КОДЕКА"),
        ("POINTS", "ТОЧЕК"),
        ("Validation points", "Точек проверки"),
        ("Valid BEFORE", "Корректных ДО"),
        ("Valid AFTER", "Корректных ПОСЛЕ"),
        ("SOURCE TARGET", "ИСХОДНАЯ ЦЕЛЕВАЯ ДОРОЖКА"),
        ("REFERENCE DURATION", "ДЛИТЕЛЬНОСТЬ ЭТАЛОНА"),
        ("TARGET ORIGINAL DURATION", "ИСХОДНАЯ ДЛИТЕЛЬНОСТЬ ЦЕЛЕВОЙ ДОРОЖКИ"),
        ("OUTPUT DURATION", "ДЛИТЕЛЬНОСТЬ ВЫХОДА"),
        ("OUTPUT - REFERENCE", "РАЗНИЦА ВЫХОДА С ЭТАЛОНОМ"),
        ("SOURCE CODEC", "ИСХОДНЫЙ КОДЕК"),
        ("OUTPUT ENCODER", "КОДЕК ВЫХОДА"),
        ("OUTPUT CONTAINER", "КОНТЕЙНЕР ВЫХОДА"),
        ("PCM LOSSLESS FALLBACK", "ЗАПАСНОЙ ВЫХОД PCM БЕЗ ПОТЕРЬ"),
        ("FORMAT DECISION", "РЕШЕНИЕ ПО ФОРМАТУ"),
        ("SEGMENTS", "СЕГМЕНТОВ"),
        ("CONCAT LATENCY COMPENSATION", "КОМПЕНСАЦИЯ ЗАДЕРЖКИ ОБЪЕДИНЕНИЯ"),
        ("MAX MAP APPROXIMATION ERROR", "МАКСИМАЛЬНАЯ ОШИБКА ПРИБЛИЖЕНИЯ КАРТЫ"),
        ("FILTERGRAPH", "ГРАФ ФИЛЬТРОВ"),
        ("DURATION CHECK", "ПРОВЕРКА ДЛИТЕЛЬНОСТИ"),
        ("DURATION INTERPRETATION", "ТОЛКОВАНИЕ ДЛИТЕЛЬНОСТИ"),
        ("MODE", "РЕЖИМ"),
        ("ALREADY SYNCHRONIZED", "УЖЕ СИНХРОННО"),
        ("BASELINE MEDIAN ERROR", "МЕДИАННАЯ ОШИБКА ИСХОДНОЙ ДОРОЖКИ"),
        ("BASELINE P95 ERROR", "ОШИБКА P95 ИСХОДНОЙ ДОРОЖКИ"),
        ("CANDIDATE MAX ABS OFFSET", "МАКСИМАЛЬНЫЙ МОДУЛЬ СМЕЩЕНИЯ КАРТЫ-КАНДИДАТА"),
        ("SAFE MAP MAX ABS OFFSET", "МАКСИМАЛЬНЫЙ МОДУЛЬ СМЕЩЕНИЯ БЕЗОПАСНОЙ КАРТЫ"),
        ("ACCEPTED VALIDATION POINTS", "ПРИНЯТЫХ ТОЧЕК ПРОВЕРКИ"),
        ("REJECTED VALIDATION POINTS", "ОТКЛОНЁННЫХ ТОЧЕК ПРОВЕРКИ"),
        ("TOTAL VALIDATION POINTS", "ВСЕГО ТОЧЕК ПРОВЕРКИ"),
        ("ACCEPTED CONTIGUOUS RUNS", "ПРИНЯТЫХ НЕПРЕРЫВНЫХ УЧАСТКОВ"),
        ("MINIMUM ACCEPTED RUN POINTS", "МИНИМУМ ТОЧЕК В ПРИНЯТОМ УЧАСТКЕ"),
        ("MINIMUM ACCEPTED RUN", "МИНИМАЛЬНЫЙ ПРИНЯТЫЙ УЧАСТОК"),
        ("REPAIR PASSES", "ПРОХОДОВ ЛОКАЛЬНОЙ ЗАЩИТЫ"),
        ("FORCED LOCAL ROLLBACKS", "ПРИНУДИТЕЛЬНЫХ ЛОКАЛЬНЫХ ОТКАТОВ"),
        ("LOCAL QUARANTINE CENTERS", "ЦЕНТРОВ ЛОКАЛЬНОГО КАРАНТИНА"),
        ("QUARANTINE CENTERS (s)", "ЦЕНТРЫ КАРАНТИНА (с)"),
        ("QUARANTINE CENTERS", "ЦЕНТРОВ ЛОКАЛЬНОГО КАРАНТИНА"),
        ("QUARANTINE RADIUS", "РАДИУСА ЛОКАЛЬНОГО КАРАНТИНА"),
        ("QUARANTINED VALIDATION POINTS", "ТОЧЕК ПРОВЕРКИ В КАРАНТИНЕ"),
        ("QUARANTINE FRACTION", "ДОЛИ ТОЧЕК В КАРАНТИНЕ"),
        ("WORSENED POINTS OUTSIDE QUARANTINE", "УХУДШЕННЫХ ТОЧЕК ВНЕ КАРАНТИНА"),
        ("QUARANTINED POINTS EXCLUDED FROM DECISION", "ТОЧЕК В КАРАНТИНЕ, ИСКЛЮЧЁННЫХ ИЗ РЕШЕНИЯ"),
        ("THRESHOLDS", "ПОРОГИ"),
        ("Non-regression tolerance", "Допуск без ухудшения"),
        ("Clear improvement", "Явное улучшение"),
        ("Already synchronized median threshold", "Порог медианной ошибки уже синхронной дорожки"),
        ("Zero-map threshold", "Порог нулевой карты"),
        ("Dense interval", "Интервал плотной проверки"),
        ("DENSE VALIDATION INTERVAL", "ИНТЕРВАЛ ПЛОТНОЙ ПРОВЕРКИ"),
        ("SAME GRID FOR CANDIDATE AND SAFE", "ОДНА И ТА ЖЕ СЕТКА ДЛЯ КАРТЫ-КАНДИДАТА И БЕЗОПАСНОЙ КАРТЫ"),
        ("CANDIDATE VALIDATION", "ПРОВЕРКА КАРТЫ-КАНДИДАТА"),
        ("SAFE MAP VALIDATION", "ПРОВЕРКА БЕЗОПАСНОЙ КАРТЫ"),
        ("Mean before", "Средняя ошибка до коррекции"),
        ("Mean candidate after", "Средняя ошибка после карты-кандидата"),
        ("Median before", "Медианная ошибка до коррекции"),
        ("Median candidate after", "Медианная ошибка после карты-кандидата"),
        ("Mean safe after", "Средняя ошибка после безопасной карты"),
        ("Median safe after", "Медианная ошибка после безопасной карты"),
        ("FINAL ASSESSMENT", "ИТОГОВАЯ ОЦЕНКА"),
        ("PASSED", "ПРОЙДЕНО"),
        ("REASON", "ПРИЧИНА"),
        ("WORSENED POINTS", "УХУДШЕННЫХ ТОЧЕК"),
        ("MINIMUM MEANINGFUL CORRECTION", "МИНИМАЛЬНАЯ ЗНАЧИМАЯ КОРРЕКЦИЯ"),
        ("MINIMUM MEANINGFUL IMPROVEMENT", "МИНИМАЛЬНОЕ ЗНАЧИМОЕ УЛУЧШЕНИЕ"),
        ("NOTE", "ПРИМЕЧАНИЕ"),
        ("INTERPRETATION", "ИНТЕРПРЕТАЦИЯ"),
        ("EXACT COPY", "ТОЧНАЯ КОПИЯ"),
        ("MAP", "КАРТА"),
        ("DRIFT", "ДРЕЙФ"),
        ("WARNINGS", "ПРЕДУПРЕЖДЕНИЯ"),
        ("PROBLEMS", "ПРОБЛЕМЫ"),
        ("Coverage", "Покрытие"),
        ("Median spacing", "Медианный интервал"),
        ("Maximum gap", "Максимальный пробел"),
        ("Mean slope", "Средний наклон"),
        ("Min slope", "Минимальный наклон"),
        ("Max slope", "Максимальный наклон"),
        ("Max abs slope", "Максимальный абсолютный наклон"),
        ("Max curvature", "Максимальная кривизна"),
        ("Score", "Оценка"),
        ("Median map error", "Медианная ошибка карты"),
        ("Max map error", "Максимальная ошибка карты"),
        ("Drift graph", "График дрейфа"),
        ("Validation graph", "График проверки"),
        ("Error graph", "График ошибки"),
        ("Map looks good", "Карта выглядит хорошо"),
        ("The map looks good", "Карта выглядит хорошо"),
        ("Confidence-aware analysis completed", "Анализ с учётом надёжности измерений завершён"),
        ("Confidence-aware analysis", "Анализ с учётом надёжности измерений"),
        ("Confidence-aware map", "Карта с учётом надёжности измерений"),
        ("Confidence classification", "Классификация надёжности измерений"),
        ("Global alignment complete", "Глобальное совмещение завершено"),
        ("Local anchor search complete", "Локальный поиск якорей завершён"),
        ("Local anchor search", "Локальный поиск якорей"),
        ("GRAPHICAL REPORTS", "ГРАФИЧЕСКИЕ ОТЧЁТЫ"),
        ("ENVELOPE GRAPH", "ГРАФИК ОГИБАЮЩЕЙ"),
        ("GLOBAL ALIGNMENT GRAPH", "ГРАФИК ГЛОБАЛЬНОГО СОВМЕЩЕНИЯ"),
        ("LOCAL ANCHOR GRAPH", "ГРАФИК ЛОКАЛЬНЫХ ЯКОРЕЙ"),
        ("POINTS", "ТОЧЕК"),
        ("Start map offset", "Начальное смещение карты"),
        ("End map offset", "Конечное смещение карты"),
        ("Max absolute map correction", "Максимальный модуль коррекции карты"),
        ("The next stage", "Следующий этап"),
        ("DIFFERENCE FROM PREVIOUS VERSION", "РАЗНИЦА С ПРЕДЫДУЩЕЙ ВЕРСИЕЙ"),
        ("VALIDATION INTERVAL", "ИНТЕРВАЛ ПРОВЕРКИ"),
        ("VERDICT", "ВЕРДИКТ"),
        ("Warning outliers", "ПРЕДУПРЕЖДАЮЩИХ ВЫБРОСОВ"),
        ("Raw", "Исходных"),
        ("APPLY MAP", "КАРТА ПРИМЕНЕНИЯ"),
        ("min tempo", "минимальный темп"),
        ("max tempo", "максимальный темп"),
        ("HARD OUTLIERS", "HARD-ВЫБРОСЫ"),
        ("RENDER AWARE", "ПРОВЕРКА ПО РЕАЛЬНОМУ РЕНДЕРУ"),
        ("RENDER_AWARE", "ПРОВЕРКА_ПО_РЕАЛЬНОМУ_РЕНДЕРУ"),
        ("MICRO WAVEFORM REFINEMENT", "МИКРО-УТОЧНЕНИЕ ФОРМЫ ВОЛНЫ"),
        ("MICRO WAVEFORM", "МИКРО-АНАЛИЗ ФОРМЫ ВОЛНЫ"),
        ("MICRO CORRECTION", "МИКРОКОРРЕКЦИЯ"),
        ("MICRO MATCHES", "МИКРОСОВПАДЕНИЙ"),
        ("BASELINE SCORE", "БАЗОВАЯ ОЦЕНКА"),
        ("BEST SCORE", "ЛУЧШАЯ ОЦЕНКА"),
        ("SCORE GAIN", "ВЫИГРЫШ ОЦЕНКИ"),
        ("PEAK MARGIN", "ЗАПАС ДО ВТОРОГО ПИКА"),
        ("RMS CORRELATION", "КОРРЕЛЯЦИЯ RMS"),
        ("EDGE CORRELATION", "КОРРЕЛЯЦИЯ ИЗМЕНЕНИЙ"),
    ]
    for old, new in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"(?<![A-Za-z]){re.escape(old)}(?![A-Za-z])", new, text, flags=re.IGNORECASE)

    text = re.sub(r"\bScore\s*:", "Оценка:", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMedian map error\s*:", "Медианная ошибка карты:", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMax map error\s*:", "Максимальная ошибка карты:", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDrift graph\s*:", "График дрейфа:", text, flags=re.IGNORECASE)
    text = re.sub(r"\bConfidence-aware analysis completed\b", "Анализ с учётом надёжности измерений завершён", text, flags=re.IGNORECASE)
    text = re.sub(r"\bConfidence-aware analysis\b", "Анализ с учётом надёжности измерений", text, flags=re.IGNORECASE)
    text = re.sub(r"\bConfidence-aware map\b", "Карта с учётом надёжности измерений", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAnalysis complete\b", "Анализ завершён", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMap looks good\b", "Карта выглядит хорошо", text, flags=re.IGNORECASE)

    # Специальные подписи AFTER/BEFORE и проценты с единицами измерения.
    text = re.sub(r"^AFTER\s+", "ПОСЛЕ - ", text, flags=re.IGNORECASE)
    text = re.sub(r"^BEFORE\s+", "ДО - ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bWithin\s+(\d+(?:\.\d+)?)\s*ms\s*:", r"В пределах \1 мс:", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\s*ms\b", " мс", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\s*s\b", " с", text, flags=re.IGNORECASE)
    text = text.replace("points /", "точек /")
    text = re.sub(r"(?<=\d)\s+points\b", " точек", text, flags=re.IGNORECASE)
    text = re.sub(r"\bTrue\b", "ДА", text, flags=re.IGNORECASE)
    text = re.sub(r"\bFalse\b", "НЕТ", text, flags=re.IGNORECASE)
    for old, new in ((
        ("EXCELLENT", "ОТЛИЧНО"),
        ("ACCEPTABLE", "ПРИЕМЛЕМО"),
        ("GOOD", "ХОРОШО"),
        ("BAD", "ПЛОХО"),
    )):
        text = re.sub(rf"\b{old}\b", new, text, flags=re.IGNORECASE)

    # --------------------------------------------------------
    # v6.9: финальная очистка остаточного английского языка.
    # Технические обозначения кодеков, форматов, путей и AUDACITY сохраняются.
    # --------------------------------------------------------
    residual_phrases = [
        (
            "The guard compares candidate and SAFE MAP on the SAME dense independent validation grid. Rejected points are hard barriers; correction cannot interpolate through them.",
            "Защита сравнивает карту-кандидат и БЕЗОПАСНУЮ КАРТУ на одной и той же плотной независимой сетке проверки. Отклонённые точки являются жёсткими барьерами: коррекция не может интерполироваться через них.",
        ),
        (
            "Strong singleton jumps rejected",
            "Отброшенных одиночных сильных скачков",
        ),
        (
            "DURATION INTERPRETATION: encoded output duration is reported separately from synchronization residuals.",
            "ТОЛКОВАНИЕ ДЛИТЕЛЬНОСТИ: длительность закодированного выхода указана отдельно от остаточной ошибки синхронизации.",
        ),
        (
            "FORMAT DECISION: Exact copy; no re-encoding because the non-regression guard found no safe timing correction.",
            "РЕШЕНИЕ ПО ФОРМАТУ: точная копия; перекодирование не выполнялось, поскольку защита от ухудшения не обнаружила безопасной временной коррекции.",
        ),
    ]
    for old, new in residual_phrases:
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)

    # Простые одиночные слова, которые встречались в старых версиях отчётов.
    # Они применяются только после обработки фраз, чтобы не ломать связный текст.
    generic_words = [
        ("BIDIRECTIONAL", "ДВУСТОРОННЯЯ"),
        ("VALIDATION", "ПРОВЕРКА"),
        ("CLASSIFICATION", "КЛАССИФИКАЦИЯ"),
        ("CANDIDATE", "КАНДИДАТ"),
        ("CONFIRMED", "ПОДТВЕРЖДЁННЫХ"),
        ("DISCREPANCY", "РАСХОЖДЕНИЙ"),
        ("AMBIGUOUS", "НЕОДНОЗНАЧНЫХ"),
        ("WEAK", "СЛАБЫХ"),
        ("STRONG", "СИЛЬНЫХ"),
        ("ACCEPTABLE", "ПРИЕМЛЕМЫХ"),
        ("INVALID", "НЕДЕЙСТВИТЕЛЬНЫХ"),
        ("IMPORTANT", "ВАЖНО"),
        ("NOTE", "ПРИМЕЧАНИЕ"),
        ("INTERPRETATION", "ИНТЕРПРЕТАЦИЯ"),
        ("SCORE", "ОЦЕНКА"),
        ("PASS", "ПРОЙДЕНО"),
        ("FAIL", "НЕ ПРОЙДЕНО"),
        ("BEFORE", "ДО"),
        ("AFTER", "ПОСЛЕ"),
        ("MAP", "КАРТА"),
        ("SAFE", "БЕЗОПАСНАЯ"),
        ("SOURCE", "ИСХОДНЫЙ"),
        ("CODEC", "КОДЕК"),
        ("ENCODER", "КОДЕР"),
        ("CONTAINER", "КОНТЕЙНЕР"),
        ("FORMAT", "ФОРМАТ"),
        ("SEGMENTS", "СЕГМЕНТОВ"),
        ("FILTERGRAPH", "ГРАФ ФИЛЬТРОВ"),
        ("DURATION", "ДЛИТЕЛЬНОСТЬ"),
        ("ERROR", "ОШИБКА"),
        ("RESIDUAL", "ОСТАТОК"),
        ("CORRELATION", "КОРРЕЛЯЦИЯ"),
        ("COVERAGE", "ПОКРЫТИЕ"),
        ("POINTS", "ТОЧЕК"),
        ("MAXIMUM", "МАКСИМАЛЬНАЯ"),
        ("MINIMUM", "МИНИМАЛЬНЫЙ"),
        ("TOTAL", "ВСЕГО"),
        ("START", "НАЧАЛЬНОЕ"),
        ("END", "КОНЕЧНОЕ"),
        ("TIME", "ВРЕМЯ"),
        ("RENDER", "РЕНДЕР"),
        ("COPY", "КОПИРОВАНИЕ"),
        ("REPAIR", "ВОССТАНОВЛЕНИЕ"),
        ("TRUE", "ДА"),
        ("FALSE", "НЕТ"),
        ("NONE", "НЕТ"),
    ]
    for old, new in generic_words:
        text = re.sub(
            rf"(?<![A-Za-z]){re.escape(old)}(?![A-Za-z])",
            new,
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(r"\bCHECK\b", "ПРОВЕРКА", text, flags=re.IGNORECASE)
    text = re.sub(r"\bHARD-выбросов\b", "жёстких выбросов", text, flags=re.IGNORECASE)
    text = re.sub(r"\bHARD-кандидатов\b", "жёстких кандидатов", text, flags=re.IGNORECASE)
    text = re.sub(r"\bHARD-КАНДИДАТОВ\b", "ЖЁСТКИХ КАНДИДАТОВ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bHARD-РАСХОЖДЕНИЙ\b", "ЖЁСТКИХ РАСХОЖДЕНИЙ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bHARD-выбросов кандидата\b", "жёстких выбросов кандидата", text, flags=re.IGNORECASE)
    text = text.replace("ПРИЕМЛЕМО ТОЧЕК", "ПРИЕМЛЕМЫХ ТОЧЕК")

    return leading + _restore_tokens(text)


def create_russian_report_copies(output_dir: Path):
    """Создаёт отдельные *_RU.txt, не изменяя английские отчёты."""
    created=[]
    for source in sorted(output_dir.glob("*.txt")):
        if source.name.endswith(RUSSIAN_REPORT_SUFFIX):
            continue
        if source.name.lower() == "render_filtergraph.txt":
            continue
        try:
            lines=source.read_text(encoding="utf-8",errors="replace").splitlines()
        except Exception:
            continue
        target=source.with_name(source.stem+RUSSIAN_REPORT_SUFFIX)
        out_lines=[
            "РУССКАЯ КОПИЯ ОТЧЁТА",
            "="*78,
            f"Источник: {source.name}",
            "",
        ]
        out_lines.extend(_translate_report_line_to_russian(line) for line in lines)
        target.write_text("\n".join(out_lines)+"\n",encoding="utf-8")
        created.append(target)
    return created


def _percent_text(value):
    if value is None:
        return "N/A"
    return f"{float(value):.2f}%"


def save_sync_effectiveness_report(output_dir, selected, validation_metrics, final_sync_map):
    """Human-readable summary before APPLY."""
    import numpy as np
    offsets = np.asarray(final_sync_map["offset"], dtype=np.float64)
    max_abs_offset = float(np.max(np.abs(offsets))) if offsets.size else 0.0
    start_offset = float(offsets[0]) if offsets.size else 0.0
    end_offset = float(offsets[-1]) if offsets.size else 0.0

    path = output_dir / "SYNC_EFFECTIVENESS_REPORT.txt"
    with path.open("w", encoding="utf-8") as file:
        file.write("SYNC EFFECTIVENESS REPORT\n")
        file.write("=" * 78 + "\n\n")
        file.write(f"REFERENCE: {selected.reference.path}\n")
        file.write(f"TARGET: {selected.target.path}\n\n")
        file.write(f"Start map offset: {start_offset:+.6f} s\n")
        file.write(f"End map offset: {end_offset:+.6f} s\n")
        file.write(f"Maximum absolute map correction: {max_abs_offset:.6f} s\n\n")
        file.write("INDEPENDENT VALIDATION BEFORE APPLY\n")
        file.write(f"Mean error: {validation_metrics.get('before_mean_error')} s\n")
        file.write(f"Median error: {validation_metrics.get('before_median_error')} s\n")
        file.write(f"Within 5 ms: {_percent_text(validation_metrics.get('before_within_5ms_percent'))}\n")
        file.write(f"Within 10 ms: {_percent_text(validation_metrics.get('before_within_10ms_percent'))}\n")
        file.write(f"Within 20 ms: {_percent_text(validation_metrics.get('before_within_20ms_percent'))}\n")
        file.write(f"Within 50 ms: {_percent_text(validation_metrics.get('before_within_50ms_percent'))}\n")
        file.write(f"Within 100 ms: {_percent_text(validation_metrics.get('before_within_100ms_percent'))}\n")
        file.write(f"Within 250 ms: {_percent_text(validation_metrics.get('before_within_250ms_percent'))}\n\n")
        file.write("INDEPENDENT VALIDATION AFTER MAP\n")
        file.write(f"Mean residual: {validation_metrics.get('after_mean_error')} s\n")
        file.write(f"Median residual: {validation_metrics.get('after_median_error')} s\n")
        file.write(f"Within 5 ms: {_percent_text(validation_metrics.get('after_within_5ms_percent'))}\n")
        file.write(f"Within 10 ms: {_percent_text(validation_metrics.get('after_within_10ms_percent'))}\n")
        file.write(f"Within 20 ms: {_percent_text(validation_metrics.get('after_within_20ms_percent'))}\n")
        file.write(f"Within 50 ms: {_percent_text(validation_metrics.get('after_within_50ms_percent'))}\n")
        file.write(f"Within 100 ms: {_percent_text(validation_metrics.get('after_within_100ms_percent'))}\n")
        file.write(f"Within 250 ms: {_percent_text(validation_metrics.get('after_within_250ms_percent'))}\n")
        file.write(f"Mean error improvement: {_percent_text((validation_metrics.get('error_improvement_ratio') or 0.0) * 100.0)}\n\n")
        file.write("BIDIRECTIONAL CHECK\n")
        file.write(f"Confirmed map: {validation_metrics.get('bidir_confirmed_map',0)}\n")
        file.write(f"Confirmed discrepancy: {validation_metrics.get('bidir_confirmed_discrepancy',0)}\n")
        file.write(f"Ambiguous: {validation_metrics.get('bidir_ambiguous',0)}\n")
        file.write(f"Consistent but weak: {validation_metrics.get('bidir_consistent_but_weak',0)}\n")
        file.write(f"HARD candidates: {validation_metrics.get('bidir_hard_candidates',0)}\n")
        file.write(f"HARD candidates requiring manual review: {validation_metrics.get('bidir_hard_confirmed_discrepancy',0)}\n")
        file.write(f"Ambiguous HARD candidates: {validation_metrics.get('bidir_hard_ambiguous',0)}\n\n")
        file.write("NOTE:\n")
        file.write("Percentages describe the independent validation samples that fall within the stated residual threshold; they are not a literal percentage of audible waveform similarity.\n")
    return path


def prompt_before_apply(selected, final_sync_map, validation_metrics, codec_plan, guard_report=None):
    import numpy as np
    offsets = np.asarray(final_sync_map["offset"], dtype=np.float64)
    max_abs_offset = float(np.max(np.abs(offsets))) if offsets.size else 0.0
    practical = (
        max_abs_offset <= 0.020
        and (validation_metrics.get("before_mean_error") is None or validation_metrics.get("before_mean_error") <= 0.020)
    )

    print()
    print("=" * 78)
    print("ПОДТВЕРЖДЕНИЕ ПЕРЕД APPLY")
    print("=" * 78)
    print()
    if guard_report is not None:
        mode = guard_report.get("mode", "")
        if mode in {"NO_CHANGE", "PRESERVE_TARGET", "FALLBACK_BASELINE"}:
            print("Дорожка уже практически синхронна или новая коррекция не прошла защитную проверку.")
            print("Новая карта не будет ухудшать исходный TARGET.")
        elif mode in {"MIXED_SAFE_MAP", "MIXED_SAFE_MAP_DENSE"}:
            print("Применяется защищённая карта: подтверждённые улучшения сохранены, потенциально ухудшающие участки подавлены.")
        elif mode == "LOCAL_QUARANTINE_SAFE_MAP":
            print("Применяется локально защищённая карта: подтверждённые улучшения сохранены, а конфликтующие участки переведены в локальный карантин без отмены остальных улучшений.")
        elif mode == "FULL_CANDIDATE":
            print("Карта прошла проверку без ухудшения без подавления коррекции.")
        else:
            print("Карта прошла независимую проверку и подготовлена к APPLY.")
    elif practical:
        print("Дорожки по результатам независимой проверки практически синхронны.")
    else:
        print("Обнаружен измеримый рассинхрон, карта предлагает временную коррекцию.")
    print(f"Максимальная коррекция карты : {max_abs_offset * 1000.0:.1f} мс")
    print(f"Начальный offset             : {float(offsets[0]) if offsets.size else 0.0:+.6f} s")
    print(f"Конечный offset              : {float(offsets[-1]) if offsets.size else 0.0:+.6f} s")
    print(f"Исходная средняя ошибка      : {validation_metrics.get('before_mean_error')} s")
    print(f"После карты средняя ошибка   : {validation_metrics.get('after_mean_error')} s")
    print(f"Исходная точность ±50 ms     : {_percent_text(validation_metrics.get('before_within_50ms_percent'))}")
    print(f"После карты точность ±50 ms  : {_percent_text(validation_metrics.get('after_within_50ms_percent'))}")
    print()
    print(f"TARGET codec                 : {selected.target.codec_name}")
    print(f"Выходной encoder             : {codec_plan.get('encoder')}")

    if codec_plan.get("pcm_fallback"):
        print()
        print("ВАЖНО: выбран lossless PCM fallback.")
        print("Причина:")
        print(codec_plan.get("reason"))
        print("Выход будет WAV/RF64, все каналы и sample rate сохраняются.")

    print()
    print("Применить карту и создать новый синхронизированный файл? [Y/N]")
    try:
        answer = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    return answer in {"y", "yes", "д", "да"}


def save_final_effectiveness_report(
    output_dir,
    selected,
    before_metrics,
    final_metrics,
    final_verification,
):
    path = output_dir / "FINAL_SYNC_EFFECTIVENESS_REPORT.txt"
    before_mean = before_metrics.get("before_mean_error")
    final_mean = final_metrics.get("after_mean_error")
    if before_mean is not None and final_mean is not None and before_mean > 1e-12:
        improvement = (1.0 - float(final_mean) / float(before_mean)) * 100.0
    else:
        improvement = None

    with path.open("w", encoding="utf-8") as file:
        file.write("FINAL SYNC EFFECTIVENESS REPORT\n")
        file.write("=" * 78 + "\n\n")
        file.write(f"REFERENCE DURATION: {selected.reference.duration:.6f} s\n")
        file.write(f"TARGET ORIGINAL DURATION: {selected.target.duration:.6f} s\n")
        file.write(f"OUTPUT DURATION: {final_verification['output_info'].duration:.6f} s\n")
        file.write(f"OUTPUT - REFERENCE: {final_verification['duration_difference']:+.6f} s\n")
        file.write(f"DURATION CHECK: {'PASS' if final_verification['duration_ok'] else 'FAIL'}\n")
        file.write("DURATION INTERPRETATION: encoded output duration is reported separately from synchronization residuals.\n")
        diag=final_verification.get('duration_diagnostics',{})
        if diag.get('codec_frame_duration_sec') is not None:
            file.write(f"KNOWN CODEC FRAME DURATION: {diag['codec_frame_duration_sec']:.6f} s\n")
            file.write(f"NEAREST CODEC FRAME MULTIPLE: {diag.get('nearest_frame_count')}\n")
            file.write(f"FRAME-GRANULARITY RESIDUAL: {diag.get('frame_granularity_residual_sec'):+.6f} s\n")
        file.write("\n")
        file.write("BEFORE APPLY\n")
        file.write(f"Mean error: {before_metrics.get('before_mean_error')} s\n")
        file.write(f"Median error: {before_metrics.get('before_median_error')} s\n")
        file.write(f"Within 20 ms: {_percent_text(before_metrics.get('before_within_20ms_percent'))}\n")
        file.write(f"Within 50 ms: {_percent_text(before_metrics.get('before_within_50ms_percent'))}\n")
        file.write(f"Within 100 ms: {_percent_text(before_metrics.get('before_within_100ms_percent'))}\n\n")
        file.write("FINAL RENDER VALIDATION\n")
        file.write(f"Mean residual: {final_metrics.get('after_mean_error')} s\n")
        file.write(f"Median residual: {final_metrics.get('after_median_error')} s\n")
        file.write(f"Max residual: {final_metrics.get('after_max_error')} s\n")
        file.write(f"Within 5 ms: {_percent_text(final_metrics.get('after_within_5ms_percent'))}\n")
        file.write(f"Within 10 ms: {_percent_text(final_metrics.get('after_within_10ms_percent'))}\n")
        file.write(f"Within 20 ms: {_percent_text(final_metrics.get('after_within_20ms_percent'))}\n")
        file.write(f"Within 50 ms: {_percent_text(final_metrics.get('after_within_50ms_percent'))}\n")
        file.write(f"Within 100 ms: {_percent_text(final_metrics.get('after_within_100ms_percent'))}\n")
        file.write(f"Within 250 ms: {_percent_text(final_metrics.get('after_within_250ms_percent'))}\n")
        file.write(f"Residual reduction: {_percent_text(improvement)}\n")
        file.write(f"Validation outlier candidates: {len(final_verification['outliers'])}\n")
        file.write(f"Confirmed discrepancies: {final_metrics.get('bidir_confirmed_discrepancy',0)}\n")
        file.write(f"Ambiguous validation points: {final_metrics.get('bidir_ambiguous',0)}\n")
        file.write(f"HARD candidates requiring manual review: {final_metrics.get('bidir_hard_confirmed_discrepancy',0)}\n")
        file.write(f"Ambiguous HARD candidates: {final_metrics.get('bidir_hard_ambiguous',0)}\n\n")
        file.write("INTERPRETATION:\n")
        file.write("The threshold percentages describe the share of independent validation samples inside each residual band. They are not a literal percentage of perceived audio quality.\n")
    return path
