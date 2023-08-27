from lgmcts.components.conception import Chatter

class LLM(Chatter):
    """Abstract class for large language model"""

    def __init__(self):
        self._is_api = False

    @property
    def is_api(self):
        return self._is_api

    @is_api.setter
    def is_api(self, is_api):
        self._is_api = is_api
