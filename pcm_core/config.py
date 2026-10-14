from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    analysis_rate: int = 48000
    anchor_step: float = 5.0
    micro_radius: float = 0.025
    micro_window: float = 0.22
    min_raw_score: float = 0.55
    min_raw_margin: float = 0.06
    raw_consensus: float = 0.00030
    min_improvement: float = 0.25 / 48000
    max_rate_change: float = 0.08
    sinc_radius: int = 16

    def __post_init__(self):
        if not 0.5 <= self.anchor_step <= 30:
            raise ValueError("Шаг анализа должен быть от 0,5 до 30 секунд.")
        if self.analysis_rate < 32000 or not .001 <= self.micro_radius <= .1:
            raise ValueError('Некорректные параметры PCM-микропоиска.')
