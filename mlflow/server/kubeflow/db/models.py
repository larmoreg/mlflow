from sqlalchemy import Column, String, Integer, UniqueConstraint
from sqlalchemy.orm import declarative_base
from mlflow.server.kubeflow.entities import ExperimentPermission, RegisteredModelPermission

Base = declarative_base()


class SqlExperimentPermission(Base):
    __tablename__ = "experiment_permissions"
    id = Column(Integer(), primary_key=True)
    experiment_id = Column(String(255), nullable=False)
    namespace = Column(String(255), nullable=False)
    __table_args__ = (
        UniqueConstraint("experiment_id", "namespace", name="unique_experiment_namespace"),
    )

    def to_mlflow_entity(self):
        return ExperimentPermission(experiment_id=self.experiment_id, namespace=self.namespace)


class SqlRegisteredModelPermission(Base):
    __tablename__ = "registered_model_permissions"
    id = Column(Integer(), primary_key=True)
    name = Column(String(255), nullable=False)
    namespace = Column(String(255), nullable=False)
    __table_args__ = (UniqueConstraint("name", "namespace", name="unique_name_namespace"),)

    def to_mlflow_entity(self):
        return RegisteredModelPermission(name=self.name, namespace=self.namespace)
