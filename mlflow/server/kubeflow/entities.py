class ExperimentPermission:
    def __init__(self, experiment_id, namespace):
        self._experiment_id = experiment_id
        self._namespace = namespace

    @property
    def experiment_id(self):
        return self._experiment_id

    @property
    def namespace(self):
        return self._namespace

    def to_json(self):
        return {"experiment_id": self.experiment_id, "namespace": self.namespace}

    @classmethod
    def from_json(cls, dictionary):
        return cls(experiment_id=dictionary["experiment_id"], namespace=dictionary["namespace"])


class RegisteredModelPermission:
    def __init__(self, name, namespace):
        self._name = name
        self._namespace = namespace

    @property
    def name(self):
        return self._name

    @property
    def namespace(self):
        return self._namespace

    def to_json(self):
        return {"name": self.name, "namespace": self.namespace}

    @classmethod
    def from_json(cls, dictionary):
        return cls(name=dictionary["name"], namespace=dictionary["namespace"])
