from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    analysis_rate: int = 48000
    envelope_hop: float = 0.005
    envelope_window: float = 0.020
    anchor_step: float = 5.0
    search_radius: float = 120.0
    coarse_window: float = 6.0
    micro_radius: float = 0.025
    micro_window: float = 0.22
    min_coarse_score: float = 0.55
    min_coarse_margin: float = 0.04
    min_raw_score: float = 0.55
    min_raw_margin: float = 0.06
    raw_consensus: float = 0.00030
    min_improvement: float = 0.25 / 48000
    max_rate_change: float = 0.08
    max_untrusted_gap: float = 20.0
    sinc_radius: int = 16
    max_micro_plots: int = 24

    def __post_init__(self):
        if not 0.5 <= self.anchor_step <= 30:
            raise ValueError("Шаг анализа должен быть от 0,5 до 30 секунд.")
        if not 0.1 <= self.search_radius <= 7200:
            raise ValueError("Радиус поиска должен быть от 0,1 до 7200 секунд.")
