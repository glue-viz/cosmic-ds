import json
import requests

from echo import add_callback
import ipyvuetify as v

from cosmicds.components import Table
from cosmicds.phases import Stage
from cosmicds.utils import API_URL, CDSJSONEncoder

class HubbleStage(Stage):

    _measurement_mapping = {
        "measwave": "obs_wave_value",
        "restwave": "rest_wave_value",
        "velocity": "velocity_value",
        "distance": "est_dist_value",
        "name": "galaxy_name",
        "student_id": "student_id",
        "angular_size" : "ang_size_value"
    }

    _units = {
        "rest_wave_unit": "angstrom",
        "obs_wave_unit": "angstrom",
        "est_dist_unit": "Mpc",
        "velocity_unit": "km / s",
        "ang_size_unit": "arcsecond"
    }

    def __init__(self, session, story_state, app_state, *args, **kwargs):
        super().__init__(session, story_state, app_state, *args, **kwargs)

        # Respond to dark/light mode change
        add_callback(self.app_state, 'dark_mode', self._on_dark_mode_change)

    
    @classmethod
    def _map_key(cls, key):
        return cls._measurement_mapping.get(key, key)

    def _prepare_measurement(self, measurement):
        prepared = { HubbleStage._map_key(k) : measurement.get(k, None) for k in HubbleStage._measurement_mapping.keys() }
        prepared.update(HubbleStage._units)
        prepared["student_id"] = self.app_state.student["id"]
        prepared = json.loads(json.dumps(prepared, cls=CDSJSONEncoder))
        return prepared

    def submit_measurement(self, measurement):
        prepared = self._prepare_measurement(measurement)
        requests.put(f"{API_URL}/submit-measurement", json=prepared)

    def update_data_value(self, dc_name, comp_name, value, index):
        super().update_data_value(dc_name, comp_name, value, index)

        if self.app_state.update_db \
            and dc_name == "student_measurements" \
            and comp_name in HubbleStage._measurement_mapping.keys():

            data = self.data_collection[dc_name]
            measurement = { comp.label: data[comp][index] for comp in data.main_components }
            self.submit_measurement(measurement)

    def add_data_values(self, dc_name, values):
        super().add_data_values(dc_name, values)

        if self.app_state.update_db and dc_name == "student_measurements":
            self.submit_measurement(values)

    def table_selected_color(self, dark):
        theme = v.theme.themes.dark if dark else v.theme.themes.light
        return theme.info

    def _on_dark_mode_change(self, dark):
        color = self.table_selected_color(dark)
        for widget in self.widgets.values():
            if isinstance(widget, Table):
                widget.selected_color = color
