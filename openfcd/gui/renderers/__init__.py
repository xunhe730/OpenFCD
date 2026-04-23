from openfcd.core.figures import RENDERERS
from openfcd.gui.renderers.eta_heatmap import EtaHeatmapRenderer
from openfcd.gui.renderers.wavelength_profile import WavelengthProfileRenderer
from openfcd.gui.renderers.rms_map import RmsMapRenderer
from openfcd.gui.renderers.sample_grid import SampleGridRenderer
from openfcd.gui.renderers.eta_timeseries import EtaTimeseriesRenderer

RENDERERS["eta_heatmap"] = EtaHeatmapRenderer()
RENDERERS["wavelength_profile"] = WavelengthProfileRenderer()
RENDERERS["rms_map"] = RmsMapRenderer()
RENDERERS["sample_grid"] = SampleGridRenderer()
RENDERERS["eta_timeseries"] = EtaTimeseriesRenderer()

__all__ = ["RENDERERS"]
