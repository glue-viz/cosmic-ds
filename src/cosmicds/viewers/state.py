from math import ceil, floor

from echo import add_callback, callback_property, delay_callback, CallbackProperty
from glue.core import Subset
from glue.viewers.histogram.state import HistogramViewerState
from glue.viewers.scatter.state import ScatterViewerState
from glue_plotly.viewers.histogram.state import PlotlyHistogramViewerState
from numpy import linspace, inf, isfinite, isnan, spacing

from cosmicds.utils import frexp10, DEFAULT_VIEWER_HEIGHT


def cds_viewer_state(state_class):

    class CDSViewerState(state_class):

        TICK_SPACINGS = [10, 7.5, 5, 4, 3, 2.5, 2, 1]

        xtick_values = CallbackProperty([])
        ytick_values = CallbackProperty([])

        viewer_height = CallbackProperty(DEFAULT_VIEWER_HEIGHT)
        viewer_width = CallbackProperty(400)

        subtitle = CallbackProperty("")

        reset_limits_from_visible = CallbackProperty(True)

        @staticmethod
        def tick_spacing(naive_spacing):
            mantissa, exp = frexp10(naive_spacing)
            frac = CDSViewerState.best_spacing_frac(mantissa)
            return round(frac * (10 ** exp))

        @classmethod
        def best_spacing_frac(cls, frac):
            default = (-1, cls.TICK_SPACINGS[-1])
            index, fless = next(((i, t) for i, t in enumerate(cls.TICK_SPACINGS) if frac >= t), default)
            fmore = cls.TICK_SPACINGS[index - 1]
            dist_less = abs(frac - fless)
            dist_more = abs(frac - fmore)
            spacing = fless if dist_less < dist_more else fmore
            return spacing

        @callback_property
        def nxticks(self):
            return self._nxticks

        @nxticks.setter
        def nxticks(self, nticks):
            if nticks == self._nxticks:
                return
            self._nxticks = max(nticks, 1)
            self.update_xticks()

        @callback_property
        def nyticks(self):
            return self._nyticks

        @nyticks.setter
        def nyticks(self, nticks):
            if nticks == self._nyticks:
                return
            self._nyticks = max(nticks, 1)
            self.update_yticks()

        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._nxticks = 7
            self._nyticks = 7
            add_callback(self, "x_min", self._update_xmin)
            add_callback(self, "x_max", self._update_xmax)
            add_callback(self, "y_min", self._update_ymin)
            add_callback(self, "y_max", self._update_ymax)

        def _update_xmin(self, value):
            self.update_xticks(xmin=value)

        def _update_xmax(self, value):
            self.update_xticks(xmax=value)

        def _update_ymin(self, value):
            self.update_yticks(ymin=value)

        def _update_ymax(self, value):
            self.update_yticks(ymax=value)

        def update_xticks(self, xmin=None, xmax=None):
            xmin = xmin or self.x_min
            xmax = xmax or self.x_max
            if xmin is None or xmax is None:
                return
            x_range = xmax - xmin
            if isnan(x_range):
                x_range = 1
            naive = ceil(x_range / self.nxticks)
            spacing = CDSViewerState.tick_spacing(naive) if naive > 0 else 1
            self.set_xtick_spacing(spacing)

        def update_yticks(self, ymin=None, ymax=None):
            ymin = ymin or self.y_min
            ymax = ymax or self.y_max
            if ymin is None or ymax is None:
                return
            y_range = ymax - ymin
            if isnan(y_range):
                y_range = 1
            naive = ceil(y_range / self.nyticks)
            spacing = CDSViewerState.tick_spacing(naive) if naive > 0 else 1
            self.set_ytick_spacing(spacing)

        def set_xtick_spacing(self, spacing):
            xmin = 0 if isnan(self.x_min) else self.x_min
            xmax = 1 if isnan(self.x_max) else self.x_max
            tmin = ceil(xmin / spacing) * spacing
            tmax = floor(xmax / spacing) * spacing
            n = int(abs(tmax - tmin) / spacing) + 1
            self.xtick_values = list(linspace(tmin, tmax, n))

        def set_ytick_spacing(self, spacing):
            ymin = 0 if isnan(self.y_min) else self.y_min
            ymax = 1 if isnan(self.y_max) else self.y_max
            tmin = ceil(ymin / spacing) * spacing
            tmax = floor(ymax / spacing) * spacing
            n = int(abs(tmax - tmin) / spacing) + 1
            self.ytick_values = list(linspace(tmin, tmax, n))

    return CDSViewerState


class CDSScatterViewerState(ScatterViewerState):

    def _layer_bounds(self, layer_state, att):
        data = layer_state.layer
        if isinstance(data, Subset):
            subset = data
            data = subset.data
            subset_state = subset.subset_state
        else:
            subset_state = None

        min_value = data.compute_statistic('minimum', att, subset_state=subset_state)
        max_value = data.compute_statistic('maximum', att, subset_state=subset_state)
        
        rng = max_value - min_value
        margin = 0.04  # Default glue scatter viewer value
        min_value -= rng * margin
        max_value += rng * margin
        return min_value, max_value
    
    def _bounds_for_att(self, att):
        bounds = [self._layer_bounds(layer, att) for layer in filter(lambda layer: layer.visible, self.layers)]
        min_value = min((b[0] for b in bounds), default=0)
        max_value = max((b[1] for b in bounds), default=1)
        return min_value, max_value
    
    def reset_limits(self, visible_only=None):
        if visible_only is None:
            visible_only = getattr(self, "reset_limits_from_visible", True)
        if not visible_only:
            super().reset_limits()
            return

        self.reset_x_limits()
        self.reset_y_limits()

    def reset_x_limits(self):
        x_min, x_max = self._bounds_for_att(self.x_att)
        with delay_callback(self, 'x_min', 'x_max'):
            self.x_min = x_min
            self.x_max = x_max

    def reset_y_limits(self):
        y_min, y_max = self._bounds_for_att(self.y_att)
        with delay_callback(self, 'y_min', 'y_max'):
            self.y_min = y_min
            self.y_max = y_max



class CDSHistogramViewerState(PlotlyHistogramViewerState):

    gaps = CallbackProperty(True)
    gap_fraction = CallbackProperty(0.15)

    def _reset_x_limits(self):
        bounds = []
        for layer in filter(lambda layer: layer.visible, self.layers):
            data = layer.layer
            if isinstance(data, Subset):
                subset = data
                data = subset.data
                subset_state = subset.subset_state
            else:
                subset_state = None

            min_value = data.compute_statistic('minimum', self.x_att, subset_state=subset_state)
            max_value = data.compute_statistic('maximum', self.x_att, subset_state=subset_state)
            bounds.append([min_value, max_value])
        
        min_value = min((b[0] for b in bounds), default=0)
        max_value = max((b[1] for b in bounds), default=1)

        with delay_callback(self, 'x_min', 'x_max'):
            self.x_min = min_value
            self.x_max = max_value - spacing(max_value)
    
    def reset_limits(self, visible_only=None):
        if visible_only is None:
            visible_only = getattr(self, "reset_limits_from_visible", True)
        if not visible_only:
            super().reset_limits()
            return

        self._reset_x_limits()
        y_min = min(getattr(layer, '_y_min', inf) for layer in self.layers if layer.visible)
        if isfinite(y_min):
            self.y_min = y_min
        y_max = max(getattr(layer, '_y_max', 0) for layer in self.layers if layer.visible)
        if isfinite(y_max):
            self.y_max = y_max
