"""English terminal presentation. Stored reports and processing data stay untouched."""
import builtins
from contextlib import contextmanager
from functools import wraps
import re
import sys


CATALOG = {
    'МИКРО-УТОЧНЕНИЕ: локальная проверка формы волны с детализацией': 'MICRO-REFINEMENT: local waveform resolution',
    'Сравнение:': 'Comparison:', 'окна с шагом': 'windows; step', 'вокруг уже найденной карты.': 'around the current time map.',
    'Плотная защита: независимая проверка каждые': 'Dense regression check: independent interval',
    'одна и та же сетка для candidate и SAFE MAP': 'shared grid for candidate and SAFE MAP',
    'Локальная защита сработала: подтверждённые улучшения сохранены, а': 'Local regression check: verified improvements retained;',
    'конфликтных участков оставлены без коррекции.': 'conflict regions left unchanged.',
    'Применяется локально защищённая карта: подтверждённые улучшения сохранены, а конфликтующие участки переведены в локальный карантин без отмены остальных улучшений.': 'Rendering a locally verified map: retained improvements are protected; conflicting regions are excluded.',
    'СОХРАННОСТЬ БАЗОВОГО РЕЗУЛЬТАТА': 'RENDER VALIDATION: BASELINE REGRESSION CHECK',
    'ПРОВЕРКА НА ОДИНАКОВОМ ВРЕМЕНИ ЭТАЛОНА': 'RENDER VALIDATION: SHARED REFERENCE TIMELINE',
    'НЕЗАВИСИМАЯ ПРОВЕРКА МИКРОПОПРАВОК': 'PCM VALIDATION: INDEPENDENT CHECKPOINTS',
    'ПОИСК ОСТАТОЧНОГО СДВИГА ПО PCM-ОТСЧЁТАМ': 'PCM ANALYSIS: RESIDUAL DELAY',
    'БАЗОВЫЙ ВАРИАНТ': 'BASELINE PLAN', 'УТОЧНЁННЫЙ ВАРИАНТ': 'REFINED PLAN',
    'ПРОВЕРКА БАЗОВОГО PCM': 'BASELINE PCM VALIDATION',
    'ТОЧЕЧНОЕ УТОЧНЕНИЕ ОСТАВШИХСЯ ПЕРЕХОДОВ': 'PCM REFINEMENT: REMAINING TRANSITIONS',
    'НЕЗАВИСИМАЯ ПРОВЕРКА ТОЧЕЧНЫХ ПОПРАВОК': 'PCM VALIDATION: LOCAL CORRECTIONS',
    'ПРОВЕРКА PCM ПЕРЕД ЗАПИСЬЮ': 'PCM VALIDATION: BEFORE RENDER',
    'ПРОВЕРКА ОСНОВНОЙ PCM-КАРТЫ ПОСЛЕ ОТМЕНЫ УТОЧНЕНИЯ': 'PCM VALIDATION: RETAINED MAP',
    'ПРОВЕРКА БАЗОВОЙ КАРТЫ НА ОБЩЕЙ СЕТКЕ': 'BASELINE VALIDATION: SHARED GRID',
    'ИЗМЕРЕНИЕ ЛОКАЛЬНОЙ СКОРОСТИ И ЗАДЕРЖКИ PCM': 'PCM ANALYSIS: LOCAL RATE AND DELAY',
    'ПРОВЕРКА БАЗОВОЙ КАРТЫ НА НОВЫХ ТОЧКАХ PCM': 'BASELINE VALIDATION: ADDITIONAL CHECKPOINTS',
    'ВЫБОР ЛОКАЛЬНЫХ УЛУЧШЕНИЙ С СОХРАНЕНИЕМ БАЗОВОГО PCM': 'PCM SELECTION: VERIFIED LOCAL IMPROVEMENTS',
    'ПРОВЕРКА ЛОКАЛЬНЫХ PCM-СОЕДИНЕНИЙ': 'PCM VALIDATION: SEGMENT BOUNDARIES',
    'ДОИЗМЕРЕНИЕ ПРОПУЩЕННЫХ PCM-ОТРЕЗКОВ': 'PCM ANALYSIS: ADDITIONAL SEGMENTS',
    'ПРОВЕРКА ДОПОЛНИТЕЛЬНЫХ PCM-ОТРЕЗКОВ': 'PCM VALIDATION: ADDITIONAL SEGMENTS',
    'ПРОВЕРКА ОБЩЕЙ СКОРОСТИ ОТРЕЗКА': 'PCM VALIDATION: SEGMENT RATE',
    'дополнительные PCM-якоря': 'additional PCM anchors', 'сдвинутая сетка': 'offset grid',
    'СКВОЗНАЯ ПРОВЕРКА PCM': 'PCM VALIDATION: FULL PIPELINE',
    'ДЕКОДИРОВАНИЕ ВСЕХ КАНАЛОВ ДЛЯ МИКРОУТОЧНЕНИЯ': 'PCM DECODE: ALL CHANNELS',
    'ПРИМЕНЕНИЕ ПОДТВЕРЖДЁННЫХ МИКРОПОПРАВОК': 'PCM RENDER: VERIFIED CORRECTIONS',
    'ПРОВЕРКА PCM ГОТОВОГО ФАЙЛА': 'PCM VALIDATION: RENDERED OUTPUT',
    'СРАВНЕНИЕ ГОТОВОГО ФАЙЛА С ОСНОВНЫМ РЕЗУЛЬТАТОМ': 'RENDER VALIDATION: BASELINE COMPARISON',
    'УТОЧНЕНИЕ УЧАСТКОВ С МЕНЯЮЩИМСЯ ОСТАТОЧНЫМ СДВИГОМ': 'PCM REFINEMENT: VARIABLE RESIDUAL DELAY',
    'НЕЗАВИСИМАЯ ПРОВЕРКА УТОЧНЁННОЙ СЕТКИ PCM': 'PCM VALIDATION: REFINED GRID',
    'Новые локальные улучшения подтверждены; базовые улучшения сохранены.': 'New local improvements verified; baseline improvements retained.',
    'ПРОГРАММА ОСТАНОВЛЕНА ПОЛЬЗОВАТЕЛЕМ': 'CANCELLED BY USER',
    'УТОЧНЕНИЕ УЖЕ СИНХРОНИЗИРОВАННОЙ ДОРОЖКИ': 'RESIDUAL REFINEMENT OF AN ALIGNED AUDIO TRACK',
    'СКАНИРОВАНИЕ ПАПКИ': 'INPUT SCAN', 'Найдено файлов-кандидатов:': 'Candidate files:',
    'Найдено аудиодорожек:': 'Audio tracks found:', 'Найдено аудиофайлов:': 'Audio files found:',
    'Выбери эталонную дорожку': 'Select the reference audio track',
    'Выбери дорожку с рассинхроном': 'Select the audio track to synchronize',
    'Введите номер и нажмите Enter:': 'Enter a number and press Enter:',
    'Выбрано:': 'Selected:', 'ВЫБРАННЫЕ ДОРОЖКИ': 'SELECTED AUDIO TRACKS',
    'TARGET / РАССИНХРОН:': 'TARGET:', 'Начать анализ?': 'Start analysis?',
    'АНАЛИЗ ВЫБРАННОЙ ПАРЫ': 'INPUT PAIR SUMMARY',
    'неизвестна до декодирования': 'unknown until decoding',
    'будет рассчитана после декодирования': 'available after decoding',
    'РАБОЧАЯ ПАПКА': 'OUTPUT DIRECTORY', 'ПЛАН АЛГОРИТМА': 'PROCESSING PLAN',
    'Декодировать первый канал Reference.': 'Decode the first reference channel.',
    'Декодировать первый канал Target.': 'Decode the first target channel.',
    'Привести их к единой системе анализа.': 'Prepare a shared analysis format.',
    'Построить waveform / амплитудную огибающую.': 'Build waveform and amplitude-envelope data.',
    'Нормализовать громкость каждого локального участка.': 'Normalize local analysis windows.',
    'Выполнить грубое глобальное сопоставление.': 'Estimate global alignment.',
    'Разбить фильм на независимые временные окна.': 'Divide the audio into independent analysis windows.',
    'Для каждого окна искать лучшее соответствие волн.': 'Match the waveforms in each window.',
    'Проверять силу найденного совпадения.': 'Measure match confidence.',
    'Отбрасывать ложные совпадения.': 'Reject unreliable matches.',
    'Строить нелинейную временную карту.': 'Build a nonlinear time map.',
    'Сглаживать карту без уничтожения реального drift.': 'Smooth the map while retaining measured drift.',
    'Проверять монотонность времени.': 'Check time-map monotonicity.',
    'Строить графические JPEG-отчёты.': 'Create diagnostic plots.',
    'Выдавать оценку качества карты.': 'Evaluate time-map quality.',
    'Предлагать применить карту к исходному Target.': 'Request confirmation before rendering.',
    'Создать новую синхронизированную аудиодорожку.': 'Render a separate synchronized audio track.',
    'Проверить финальный файл.': 'Validate the rendered output.',
    'ПОДГОТОВКА AUDIO DATA': 'AUDIO PREPARATION', 'ДЕКОДИРОВАНИЕ REFERENCE': 'REFERENCE DECODE',
    'ДЕКОДИРОВАНИЕ TARGET': 'TARGET DECODE', 'Декодирование завершено:': 'Decoding completed:',
    'Декодирование:': 'Decoding:', 'Анализируется: первый канал': 'Analysis channel: first channel',
    'Исходный codec:': 'Input codec:', 'Исходный sample rate:': 'Input sample rate:',
    'неизвестна — будет определена после decode': 'unknown; available after decoding',
    'Длительность:': 'Duration:', 'Размер PCM': 'PCM size', 'Скорость': 'Speed',
    'Разница': 'Difference', 'duration была неизвестна до decode': 'input duration was unknown',
    'ПОСТРОЕНИЕ REFERENCE WAVEFORM': 'REFERENCE WAVEFORM', 'ПОСТРОЕНИЕ TARGET WAVEFORM': 'TARGET WAVEFORM',
    'ПОСТРОЕНИЕ REFERENCE ENVELOPE': 'REFERENCE ENVELOPE', 'ПОСТРОЕНИЕ TARGET ENVELOPE': 'TARGET ENVELOPE',
    'Огибающая готова': 'Envelope ready', 'Waveform готов': 'Waveform ready',
    'Reference waveform готов': 'Reference waveform ready', 'Target waveform готов': 'Target waveform ready',
    'Reference envelope готов': 'Reference envelope ready', 'Target envelope готов': 'Target envelope ready',
    'Рабочие данные сохранены в:': 'Analysis data saved to:', 'Реальные длительности после decode:': 'Decoded durations:',
    'ГРАФИЧЕСКИЕ ОТЧЁТЫ': 'DIAGNOSTIC PLOTS', 'ПОСТРОЕНИЕ WAVEFORM ГРАФИКА': 'WAVEFORM PLOT',
    'ПОСТРОЕНИЕ ENVELOPE ГРАФИКА': 'ENVELOPE PLOT', 'ПОСТРОЕНИЕ GLOBAL ALIGNMENT ГРАФИКА': 'GLOBAL ALIGNMENT PLOT',
    'ПОСТРОЕНИЕ LOCAL ANCHOR ГРАФИКА': 'LOCAL ANCHOR PLOT', 'График сохранён:': 'Plot saved:',
    'Графические отчёты готовы': 'Diagnostic plots ready', 'ПРЕДВАРИТЕЛЬНЫЙ ОТЧЁТ О СОВМЕЩЕНИИ': 'INITIAL ALIGNMENT SUMMARY',
    'ПРЕДВАРИТЕЛЬНАЯ ОЦЕНКА:': 'INITIAL ASSESSMENT:',
    'Большинство локальных участков имеет устойчивое совпадение.': 'Most local windows have consistent matches.',
    'Полная информация сохранена в:': 'Full details saved to:', 'ОЧИСТКА RAW ЯКОРЕЙ': 'ANCHOR FILTERING',
    'Очистка якорей завершена': 'Anchor filtering completed', 'ПОСТРОЕНИЕ НЕЛИНЕЙНОЙ КАРТЫ': 'NONLINEAR TIME MAP',
    'Нелинейная карта построена': 'Nonlinear map ready', 'СГЛАЖИВАНИЕ КАРТЫ': 'TIME-MAP SMOOTHING',
    'Карта сглажена и проверена': 'Map smoothed and checked', 'ГРАФИК КАЧЕСТВА КАРТЫ': 'TIME-MAP QUALITY PLOT',
    'Карта визуализирована:': 'Map plot saved:', 'ОЦЕНКА КАЧЕСТВА ФИНАЛЬНОЙ КАРТЫ': 'FINAL TIME-MAP QUALITY',
    'ПОСТРОЕНИЕ ФИНАЛЬНОЙ КАРТЫ': 'FINAL TIME MAP', 'Финальная карта:': 'Final map:', 'АНАЛИЗ DRIFT': 'DRIFT ANALYSIS',
    'Финальный анализ карты завершён': 'Final map analysis completed', 'ФИНАЛЬНЫЙ ВЕРДИКТ ПО КАРТЕ': 'FINAL TIME-MAP VERDICT',
    'ОЦЕНКА:': 'ASSESSMENT:', 'Карта:': 'Map:', 'начальный offset:': 'initial offset:',
    'начальный offset': 'initial offset', 'конечный offset': 'final offset', 'общий drift': 'total drift',
    'Покрытие якорями:': 'Anchor coverage:', 'ПРЕДУПРЕЖДЕНИЯ:': 'WARNINGS:',
    'В карте есть относительно резкие локальные изменения drift.': 'The map contains relatively abrupt local drift changes.',
    'Рабочая папка:': 'Output directory:', 'Финальный отчёт:': 'Final report:', 'Карта выглядит хорошо.': 'Map quality is good.',
    'На следующем этапе можно переходить к применению.': 'The map is ready for rendering.',
    'Confidence-aware analysis завершён': 'Confidence-aware analysis completed', 'ПОСТРОЕНИЕ CONFIDENCE-AWARE MAP': 'CONFIDENCE-AWARE TIME MAP',
    'Confidence-aware map готова': 'Confidence-aware map ready', 'СРАВНЕНИЕ BASE / CONFIDENCE-AWARE': 'BASE / CONFIDENCE-AWARE COMPARISON',
    'СОХРАНЕНИЕ CONFIDENCE CLASSIFICATION': 'SAVE CONFIDENCE CLASSIFICATION', 'Confidence classification сохранена': 'Confidence classification saved',
    'СОХРАНЕНИЕ CONFIDENCE-AWARE MAP': 'SAVE CONFIDENCE-AWARE MAP', 'Confidence-aware map сохранена': 'Confidence-aware map saved',
    'ГРАФИК CONFIDENCE-AWARE MAP': 'CONFIDENCE-AWARE MAP PLOT', 'Confidence-aware график готов': 'Confidence-aware plot ready',
    'НЕЗАВИСИМАЯ ПРОВЕРКА КАРТЫ': 'TIME-MAP VALIDATION: INDEPENDENT CHECKPOINTS',
    'Независимая проверка завершена': 'Independent validation completed', 'МИКРО-УТОЧНЕНИЕ ФОРМЫ ВОЛНЫ': 'WAVEFORM MICRO-REFINEMENT',
    'Проверка микросовмещения': 'Waveform micro-alignment check', 'ЗАЩИТА ОТ РЕГРЕССИЙ: DENSE GRID': 'REGRESSION CHECK: DENSE GRID',
    'Защита от регрессий завершена:': 'Regression check completed:', 'Локальный карантин': 'Local exclusion',
    'исключаются': 'excluding', 'новых проблемных точек, радиус': 'new conflict points; radius',
    'ГРАФИК BEFORE / AFTER': 'BEFORE / AFTER PLOT', 'ГРАФИК ОШИБКИ': 'ALIGNMENT ERROR PLOT',
    'СОХРАНЕНИЕ VALIDATION ТОЧЕК': 'SAVE VALIDATION CHECKPOINTS', 'Validation точки сохранены': 'Validation checkpoints saved',
    'АНАЛИЗ VALIDATION OUTLIERS': 'VALIDATION OUTLIER ANALYSIS', 'Найдено выбросов:': 'Outliers:',
    'Результаты сохранены в:': 'Results saved to:', 'ГРАФИК РАСПРЕДЕЛЕНИЯ ОШИБКИ': 'ERROR DISTRIBUTION PLOT',
    'График распределения готов': 'Distribution plot ready', 'СОХРАНЕНИЕ ROBUST MAP': 'SAVE VERIFIED MAP',
    'Robust map сохранена': 'Verified map saved', 'ПОСТРОЕНИЕ ROBUST MAP ГРАФИКА': 'VERIFIED MAP PLOT',
    'Robust map график готов': 'Verified map plot ready', 'Карта прошла строгую финальную проверку.': 'The map passed final validation.',
    'Следующий этап может переходить к APPLY.': 'Ready for rendering.',
    'Подтверждённых HARD-расхождений после двусторонней проверки не обнаружено.': 'No confirmed HARD mismatches after bidirectional validation.',
    'После основного рендера: независимая проверка остаточного сдвига по PCM.': 'After the initial render: independent PCM residual-delay validation.',
    'При PCM-уточнении сохраняются число каналов, частота и исходная разрядность PCM.': 'PCM refinement retains channel count, sample rate and PCM bit depth.',
    'Защита от ухудшения сохраняет хорошие участки без необоснованной повторной коррекции.': 'Regression checks protect already aligned segments.',
    'Подтверждённых HARD-расхождений:': 'Confirmed HARD mismatches:', 'Неоднозначных HARD-кандидатов:': 'Ambiguous HARD candidates:',
    'ПОДТВЕРЖДЕНИЕ ПЕРЕД APPLY': 'RENDER CONFIRMATION', 'Максимальная коррекция карты': 'Maximum map correction',
    'Начальный offset': 'Initial offset', 'Конечный offset': 'Final offset', 'Исходная средняя ошибка': 'Initial mean error',
    'После карты средняя ошибка': 'Mapped mean error', 'Исходная точность': 'Initial accuracy', 'После карты точность': 'Mapped accuracy',
    'Выходной encoder': 'Output encoder', 'Применить карту и создать новый синхронизированный файл?': 'Apply the map and render a separate synchronized file?',
    'Проверяем не только математическую карту, но и фактически отрендерированный сигнал.': 'Validation includes the actual rendered signal.',
    'Плотная проверка готового файла: каждые': 'Rendered-output validation interval:',
    'РЕНДЕР: ПРОХОД': 'RENDER: PASS', 'РЕНДЕР: ИСХОДНАЯ ДОРОЖКА': 'RENDER: INPUT BASELINE',
    'РЕНДЕР: ПРОВЕРКА ИСХОДНОЙ ДОРОЖКИ ПОСЛЕ ОТМЕНЫ ОСНОВНОЙ КАРТЫ': 'RENDER: INPUT BASELINE VALIDATION AFTER MAP REJECTION',
    'Проверяется вся дорожка; после отказов отменяются локальные поправки.': 'Checking the full track; rejected local corrections are removed.',
    'Участков карантина перед проходом:': 'Excluded regions before this pass:', 'Проверка реального рендера: проход': 'Rendered-signal validation: pass',
    'ПОДГОТОВКА APPLY MAP': 'RENDER MAP PREPARATION', 'Apply map подготовлена': 'Render map ready',
    'ПРИМЕНЕНИЕ ФИНАЛЬНОЙ КАРТЫ': 'AUDIO RENDER: FINAL TIME MAP', 'Render завершён:': 'Render completed:',
    'выявлено ухудшений': 'regressions found', 'Проблемные моменты:': 'Conflict checkpoints:',
    'Далее PCM-уточнение сможет независимо исправить остаточные сдвиги.': 'PCM refinement will independently evaluate residual delays.',
    'Локальное PCM-уточнение: участков': 'Local PCM refinement: segments', 'суммарно': 'total',
    'Изолировано проблемных фрагментов:': 'Excluded conflict segments:', 'Остальные поправки проверяются повторно.': 'Rechecking the remaining corrections.',
    'Сквозная проверка: улучшенных контрольных точек': 'Pipeline validation: improved checkpoints', 'отказов': 'rejections',
    'Проблемные места эталона:': 'Reference conflict checkpoints:', 'остальные — в отчёте.': 'remaining points are listed in the report.',
    'время эталона': 'reference timeline', 'сохранность базы': 'baseline regression check', 'исходник': 'input baseline',
    'СОХРАНЕНИЕ PRE-APPLY MAP': 'SAVE PRE-RENDER MAP', 'Pre-apply map сохранена': 'Pre-render map saved',
    'Apply plan сохранён': 'Render plan saved', 'ГРАФИК APPLY PLAN': 'RENDER PLAN PLOT', 'Apply plan график готов': 'Render plan plot ready',
    'ФИНАЛЬНЫЙ РЕЗУЛЬТАТ': 'FINAL RESULT', 'PCM-уточнение:': 'PCM refinement:',
    'Остаток PCM в проверенных изменённых окнах: медиана': 'PCM residual in checked modified windows: median', 'максимум': 'maximum',
    'Проверенных окон с остатком больше': 'Checked windows with residual above',
    'Оставшиеся расхождения:': 'Remaining differences:', '(если такие окна есть).': '(if any).',
    'Исходный TARGET НЕ изменён.': 'The input TARGET file is unchanged.',
    'Готовый файл создан отдельно; результат проверок приведён в отчётах.': 'Output saved separately; validation results are listed in the reports.',
    'Русских копий TXT-отчётов создано:': 'Russian TXT reports created:', 'Общее время:': 'Total elapsed:',
    'ДИАГНОСТИКА: приложите к сообщению все эти архивы:': 'DIAGNOSTICS: attach all these archives:',
    'ДИАГНОСТИКА: приложите к сообщению этот архив:': 'DIAGNOSTICS: attach this archive:',
    'Готовый файл:': 'Output file:', 'Итоговый файл:': 'Output file:', 'Отчёты:': 'Reports:',
    'Улучшение не подтверждено. Новый аудиофайл не создан.': 'No verified improvement. No new audio file was created.',
    'Исходная дорожка:': 'Input track:', 'Исходные дорожки не изменены.': 'Input tracks are unchanged.',
    'Неверный ввод.': 'Invalid input.', 'Введите корректный номер.': 'Enter a valid number.',
    'ГОТОВО': 'DONE', 'Готово': 'Done', 'ОШИБКА': 'ERROR', 'ШАГ': 'STEP',
    'оценка': 'estimate', 'проход': 'pass', 'сетка': 'grid', 'окно': 'window', 'шагом': 'step', 'шаг': 'step',
    'запись': 'render', 'мс': 'ms', 'МиБ': 'MiB', 'с.': 's.',
}

CATALOG.update({
    'ПЛОТНАЯ БЕЗОПАСНАЯ КАРТА НЕ ПРОШЛА ПРОВЕРКУ БЕЗ УХУДШЕНИЯ': 'DENSE MAP REJECTED BY REGRESSION CHECK',
    'После локального карантина сохраняем исходный TARGET без дополнительной коррекции.': 'Keeping the input TARGET after local exclusions; no additional correction.',
    'Подтверждённые HARD-расхождения:': 'Confirmed HARD mismatches:',
    'Неоднозначные HARD-кандидаты остаются только диагностикой и карту не изменяют.': 'Ambiguous HARD candidates are diagnostic only; the map is unchanged.',
    'Пользователь отменил применение карты.': 'Map application cancelled by user.',
    'Исходный TARGET не изменён.': 'The input TARGET is unchanged.',
    'PCM-уточнение завершилось ошибкой; подробности: PCM_ERROR_TRACEBACK.txt.': 'PCM refinement failed; see PCM_ERROR_TRACEBACK.txt.',
    'Анализируется: единственный канал': 'Analysis channel: mono',
    'Построение waveform': 'Building waveform', 'Построение огибающей': 'Building envelope',
    'Глобально устойчивое совпадение не найдено.': 'No consistent global match found.',
    'Совпадение есть, но карта требует дальнейшей очистки.': 'A match was found; further map filtering is required.',
    'Сейчас слишком много слабых или неоднозначных совпадений.': 'Too many weak or ambiguous matches.',
    'ДОСТУПНЫЕ АУДИОДОРОЖКИ': 'AVAILABLE AUDIO TRACKS', 'Анализ отменён.': 'Analysis cancelled.',
    'Программа ffprobe не найдена.': 'ffprobe was not found.',
    'Убедись, что FFmpeg установлен и ffprobe доступен из PATH.': 'Install FFmpeg and make ffprobe available in PATH.',
    'Нужно ввести номер.': 'Enter a number.', 'Такого номера нет.': 'Number out of range.',
    'Выбор отменён пользователем.': 'Selection cancelled by user.', 'Введите именно номер из списка.': 'Enter a number from the list.',
    'ЛОКАЛЬНАЯ ПРОВЕРКА БАЗОВЫХ ПОПРАВОК': 'BASELINE VALIDATION: LOCAL CORRECTIONS',
    'Ошибка дополнительного PCM-варианта:': 'Additional PCM plan failed:',
    'Сохранён базовый PCM-план этого запуска; далее проверяется записанный WAV.': 'Current baseline PCM plan retained; rendered WAV validation follows.',
    'ДОПОЛНИТЕЛЬНОЕ PCM-ИЗМЕРЕНИЕ У КРАЯ ФАЙЛА': 'PCM ANALYSIS: TRACK BOUNDARY',
    'ПРОВЕРКА ОБЩЕГО ОСТАТОЧНОГО СДВИГА': 'PCM VALIDATION: GLOBAL RESIDUAL DELAY',
    'Дополнительная сетка PCM отклонена; предыдущий проверенный план сохранён:': 'Additional PCM grid rejected; previous verified plan retained:',
    'Проверка сложных участков': 'Checking ambiguous segments', 'Запись confidence classification': 'Saving confidence classification',
    'Запись confidence-aware map': 'Saving confidence-aware map', 'Недостаточно dense validation-точек': 'Insufficient dense validation checkpoints',
    'Запись outliers': 'Saving outliers', 'Карта в целом рабочая, но имеет заметные локальные проблемы.': 'The map is usable but contains local conflicts.',
    'Перед APPLY желательно изучить validation_outliers.csv.': 'Review validation_outliers.csv before rendering.',
    'Карта пока НЕ готова к безопасному APPLY.': 'The map is not ready for safe rendering.',
    'Запись validation CSV': 'Saving validation CSV', 'Запись robust map': 'Saving verified map',
    'СОХРАНЕНИЕ ПРОВЕРЕННОГО РЕЗУЛЬТАТА': 'SAVE VERIFIED OUTPUT', 'Результат и отчёты сохранены': 'Output and reports saved',
    'Подтверждённых изменённых окон:': 'Verified modified windows:', 'ВАЖНО: выбран lossless PCM fallback.': 'Lossless PCM fallback selected.',
    'Выход будет WAV/RF64, все каналы и sample rate сохраняются.': 'Output: WAV/RF64; all channels and the sample rate are retained.',
    'Дорожка уже практически синхронна или новая коррекция не прошла защитную проверку.': 'The track is already aligned, or the proposed correction failed validation.',
    'Новая карта не будет ухудшать исходный TARGET.': 'The candidate must pass regression checks against the input TARGET.',
    'Дорожки по результатам независимой проверки практически синхронны.': 'Independent validation found the tracks substantially aligned.',
    'Обнаружен измеримый рассинхрон, карта предлагает временную коррекцию.': 'Measurable delay found; the map proposes a timing correction.',
    'Применяется защищённая карта: подтверждённые улучшения сохранены, потенциально ухудшающие участки подавлены.': 'Rendering a verified map: confirmed improvements retained; conflicting corrections excluded.',
    'Карта прошла проверку без ухудшения без подавления коррекции.': 'The map passed regression checks without excluding corrections.',
    'Карта прошла независимую проверку и подготовлена к APPLY.': 'The map passed independent validation and is ready to render.',
    'ДИАГНОСТИКА УКАЗАННЫХ ТОЧЕК AUDACITY': 'AUDACITY CHECKPOINT DIAGNOSTICS',
    'Создаются только короткие монофрагменты. Синхронизация фильма не запускается.': 'Exporting short mono clips only. Audio synchronization is not started.',
    'Готово. Полные дорожки и промежуточные PCM в архив не включаются.': 'Done. Full tracks and intermediate PCM are excluded from the archive.',
    'Точки через пробел (например, 7:28.755 52:05.916 52:15.301):': 'Space-separated checkpoints (example: 7:28.755 52:05.916 52:15.301):',
    'Проверка интерполированной карты': 'Checking interpolated map', 'Мягкое сглаживание': 'Map smoothing',
    'Проверка reference timeline': 'Checking reference timeline', 'СОХРАНЕНИЕ ФИНАЛЬНОЙ КАРТЫ': 'SAVE FINAL MAP',
    'Финальная карта сохранена': 'Final map saved', 'МЕТРИКИ ФИНАЛЬНОЙ КАРТЫ': 'FINAL MAP METRICS',
    'Количество raw точек': 'Raw checkpoints', 'Количество точек карты': 'Map checkpoints',
    'Общее изменение offset': 'Total offset change', 'Минимальный offset': 'Minimum offset', 'Максимальный offset': 'Maximum offset',
    'Размах offset': 'Offset range', 'Средняя скорость drift': 'Mean drift rate', 'Макс. |offset slope|': 'Max |offset slope|',
    'Макс. |offset curvature|': 'Max |offset curvature|', 'Средняя ошибка карты': 'Mean map error',
    'Медианная ошибка карты': 'Median map error', 'Максимальная ошибка карты': 'Maximum map error', 'ПРОБЛЕМЫ:': 'ISSUES:',
    'Карта выглядит приемлемо, но содержит некоторые предупреждения.': 'The map is acceptable with warnings.',
    'Следующий этап сможет применить её только после дополнительной проверки.': 'Additional validation is required before rendering.',
    'Запись final_sync_map.csv': 'Saving final_sync_map.csv', 'Карта вызывает сомнения.': 'Map confidence is low.',
    'Автоматическое применение пока не рекомендуется.': 'Automatic rendering is not recommended.', 'Карта выглядит плохой.': 'Map quality is poor.',
    'Применять её к аудио нельзя.': 'Do not apply this map to the audio.', 'Проверка локальной скорости': 'Checking local rate',
    'Запись pre_apply_sync_map.csv': 'Saving pre_apply_sync_map.csv', 'Подготовка adaptive render segments': 'Preparing adaptive render segments',
})

# Match once against the original text: replacement text is never translated
# a second time. Whole filesystem paths/file-name chunks are passed through.
_PATTERN=re.compile('|'.join(re.escape(k) for k in sorted(CATALOG,key=len,reverse=True)))
_PATH=re.compile(r'^(?:[A-Za-z]:[\\/]|/|\\\\)')
_FILE=re.compile(r'^(?:\d+\.\s*)?[^<>:"\r\n|]+\.(?:wav|wave|flac|dts|ac3|eac3|mka|mp3|m4a|ogg|opus|aac|jpg|png|zip|json|csv|txt)$',re.I)
_FILE_TAIL=re.compile(r'[^<>:"\r\n|]+\.(?:wav|wave|flac|dts|ac3|eac3|mka|mp3|m4a|ogg|opus|aac|jpg|png|zip|json|csv|txt)\s*$',re.I)


def english(text):
    text=str(text)
    if text in CATALOG:return CATALOG[text]
    if _PATH.match(text.strip()) or _FILE.fullmatch(text.strip()):return text
    tail=_FILE_TAIL.search(text)
    if tail:return english(text[:tail.start()])+tail[0]
    translated=_PATTERN.sub(lambda m:CATALOG[m[0]],text)
    translated=re.sub(r'(?<=\d) с(?=[\s.,;|)]|$)',' s',translated)
    return translated.replace(' — ',' - ').replace('…','...').replace('±','+/-')


class EnglishStream:
    _english_console=True
    def __init__(self,stream):self.stream=stream
    def write(self,text):
        self.stream.write(english(text))
        return len(text)
    def flush(self):return self.stream.flush()
    def __getattr__(self,name):return getattr(self.stream,name)


@contextmanager
def english_console():
    if getattr(sys.stdout,'_english_console',False):
        yield
        return
    original_out,original_err,original_input=sys.stdout,sys.stderr,builtins.input
    sys.stdout,sys.stderr=EnglishStream(original_out),EnglishStream(original_err)
    # input() may print its prompt through the C stdio path in a real terminal.
    builtins.input=lambda prompt='':original_input(english(prompt))
    try:yield
    finally:sys.stdout,sys.stderr,builtins.input=original_out,original_err,original_input


def english_session(function):
    @wraps(function)
    def run(*args,**kwargs):
        with english_console():return function(*args,**kwargs)
    return run
