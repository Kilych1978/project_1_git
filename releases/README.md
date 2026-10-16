# Сборка Modular 6.9.5

[Скачать ZIP](https://raw.githubusercontent.com/Kilych1978/project_1_git/refs/heads/main/releases/Universal_Audio_Waveform_Sync_modular_v6.9.5.zip)

Скачайте архив и распакуйте в новую папку. Инструкция внутри — README_RU.md.
Контрольная сумма находится рядом в файле `.zip.sha256`.

Положите две дорожки рядом со `start_sync.py`, выполните `SETUP_WINDOWS.cmd`
один раз, затем запускайте `START_WINDOWS.cmd`. Для дорожки после v3.3
с небольшим остаточным сдвигом есть `MICRO_REFINE.cmd`.
Каждый запуск работает с выбранными аудиофайлами и создаёт новые рабочие
данные. Папки, карты и отчёты от предыдущих запусков не требуются.
