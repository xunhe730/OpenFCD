from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from ruamel.yaml import YAML


class OpticalLayer(BaseModel):
    thickness_mm: float
    medium: str
    n: float


class OpticalStack(BaseModel):
    preset: Literal["pattern_below_window", "immersed_pattern", "custom"]
    layers: list[OpticalLayer] = []


class GeometryConfig(BaseModel):
    pattern_period_mm: float
    optical_stack: OpticalStack


class DataConfig(BaseModel):
    frames_dir: str
    pattern: str = "Img*.jpg"
    time_step_ms: float | None = None


class ReferenceConfig(BaseModel):
    mode: Literal["build", "use_existing", "first_frame"] = "build"
    source: str = ""
    build_params: dict = Field(default_factory=lambda: {
        "n": 100, "stride": 11, "reducer": "median"
    })


class BatchConfig(BaseModel):
    range: str
    annotation: str = ""
    forward_direction: list[float] | None = None


class MaskConfig(BaseModel):
    mode: Literal["manual", "auto", "tracked"] = "manual"
    dilate_cells: float = 3.0
    cell_mm: float = 1.2
    fallback: dict = Field(default_factory=lambda: {"mode": "manual"})


class TaperConfig(BaseModel):
    kind: Literal["tukey", "hann", "none"] = "tukey"
    alpha: float = 0.08


class ProcessConfig(BaseModel):
    flatfield_sigma: float = 300.0
    flatfield_sigma_auto: bool = True
    auto_scale_ref: bool = True
    detrend: Literal["plane", "none"] = "plane"
    taper: TaperConfig = Field(default_factory=TaperConfig)
    edge_nan_mm: float = 3.0


class ProfileConfig(BaseModel):
    px_per_mm: float = 7.27
    skip_mm: float = 5.0
    strip_mm: float = 2.0


class ROIConfig(BaseModel):
    x_mm: float = 140.0
    y_mm: float = 0.0
    auto_scale: bool = True


class RunConfig(BaseModel):
    workers: int | None = None
    chunksize: int = 16
    skip_existing: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class CompositeConfig(BaseModel):
    enabled: bool = False
    layout: Literal["1x2", "2x1", "2x2", "free"] = "2x1"
    members: list[str] = Field(default_factory=lambda: ["eta_heatmap", "wavelength_profile"])


class FiguresConfig(BaseModel):
    eta_heatmap: bool = True
    wavelength_profile: bool = True
    rms_map: bool = True
    sample_grid: bool = True
    eta_timeseries: bool = False
    composite: CompositeConfig = Field(default_factory=CompositeConfig)


class OutputConfig(BaseModel):
    dir: str = ""
    result_file: str = "results.h5"
    figures: FiguresConfig = Field(default_factory=FiguresConfig)


class VizConfig(BaseModel):
    eta_vmin_mm: float = -0.35
    eta_vmax_mm: float = 0.35
    cmap: str = "RdBu_r"
    figure_dpi: int = 200


class ProjectModel(BaseModel):
    format_version: int = 0
    name: str
    created: str
    geometry: GeometryConfig
    data: DataConfig
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    batches: list[BatchConfig] = Field(default_factory=list)
    mask: MaskConfig = Field(default_factory=MaskConfig)
    process: ProcessConfig = Field(default_factory=ProcessConfig)
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    roi: ROIConfig = Field(default_factory=ROIConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    viz: VizConfig = Field(default_factory=VizConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProjectModel:
        yaml = YAML()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.load(f)
        return cls.model_validate(dict(data))

    @classmethod
    def from_dict(cls, d: dict) -> ProjectModel:
        return cls.model_validate(d)

    def to_yaml(self, path: str | Path | None = None) -> str:
        import io
        yaml = YAML()
        yaml.default_flow_style = False
        data = self.model_dump()
        buf = io.StringIO()
        yaml.dump(data, buf)
        result = buf.getvalue()
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(result)
        return result
