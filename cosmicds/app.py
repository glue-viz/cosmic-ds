import ipyvuetify as v
import requests
import json
from echo import add_callback, ignore_callback, CallbackProperty
from glue.core.state_objects import State
from glue_jupyter.app import JupyterApplication
from glue_jupyter.state_traitlets_helpers import GlueState
from ipyvuetify import VuetifyTemplate
from ipywidgets import widget_serialization
from traitlets import Dict, Bool
from glue.core import HubListener

from .events import StepChangeMessage, WriteToDatabaseMessage
from .registries import story_registry
from .utils import load_template

from cosmicds.utils import API_URL

v.theme.dark = True

class ApplicationState(State):
    using_voila = CallbackProperty(False)
    dark_mode = CallbackProperty(True)
    student = CallbackProperty({})
    update_db = CallbackProperty(True)
    reset_student = CallbackProperty(False)

class Application(VuetifyTemplate, HubListener):
    _metadata = Dict({"mount_id": "content"}).tag(sync=True)
    story_state = GlueState().tag(sync=True)
    template = load_template("app.vue", __file__, traitlet=True).tag(sync=True)
    drawer = Bool(False).tag(sync=True)
    vue_components = Dict().tag(sync=True, **widget_serialization)
    app_state = GlueState().tag(sync=True)

    def __init__(self, story, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.app_state = ApplicationState()

        self.app_state.update_db = kwargs.get("update_db", True)
        self.team_member = kwargs.get("team_member", None)

        self._application_handler = JupyterApplication()
        self.story_state = story_registry.setup_story(story, self.session, self.app_state)

        # For testing purposes, we create a new dummy student on each startup
        student_id = kwargs.get("student_id", None)
        if student_id:
            student = { "id": student_id }
            self.student_id = student_id
            self.app_state.student = student
            self.story_state.student_user = student
            self.story_state.fetch_sample_data()
            print(f"Student id: {self.app_state.student['id']}")
        else:
            self._get_new_seed_student()

        # Initialize from database
        if self.app_state.update_db:
            self._initialize_from_database()

        # Subscribe to events
        self.hub.subscribe(self, WriteToDatabaseMessage,
                           handler=self._on_write_to_database)

        add_callback(self.app_state, 'dark_mode', self._theme_toggle)
        add_callback(self.app_state, 'reset_student', self._get_new_seed_student)

    def reload(self):
        """
        Reload only the UI elements of the application.
        """
        self.template = load_template("app.vue", __file__, traitlet=False)

    @property
    def session(self):
        """
        Underlying glue-jupyter application session instance.
        """
        return self._application_handler.session

    @property
    def data_collection(self):
        """
        Underlying glue-jupyter application data collection instance.
        """
        return self._application_handler.data_collection

    @property
    def hub(self):
        return self._application_handler.session.hub

    def _initialize_from_database(self):
        try:
            # User information for a JupyterHub notebook session is stored in an
            # environment variable
            # user = os.environ['JUPYTERHUB_USER']
            user = self.app_state.student
            story = self.story_state.name
            response = requests.get(f"{API_URL}/story-state/{user['id']}/{story}")
            data = response.json()
            state = data["state"]
            if state is not None:
                self.story_state = state
        except Exception as e:
            print(e)

    def _on_write_to_database(self, _msg):
        if not self.app_state.update_db:
            return

        # User information for a JupyterHub notebook session is stored in an
        # environment variable
        # user = os.environ['JUPYTERHUB_USER']

        user = self.app_state.student
        story = self.story_state.name
        data = json.loads(json.dumps(self.story_state.as_dict()))
        if data:
            requests.put(f"{API_URL}/story-state/{user['id']}/{story}", json=data)

    def _theme_toggle(self, dark):
        v.theme.dark = dark

    def _get_new_seed_student(self, reset=True):
        if reset and self.app_state.update_db:
            data = { "seed": True, "team_member": self.team_member }
            response = requests.post(f"{API_URL}/new-dummy-student", json=data).json()
            student = response["student"]
            self.app_state.student = student
            self.story_state.student_user = student
            self.student_id = student["id"]
            with ignore_callback(self.app_state, 'reset_student'):
                self.app_state.reset_student = False
            print(f"Student id: {self.app_state.student['id']}")
