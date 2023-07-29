from typing import List
from sqlalchemy.exc import IntegrityError, NoResultFound, MultipleResultsFound
from sqlalchemy.orm import sessionmaker

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    INVALID_STATE,
)
from mlflow.server.kubeflow.entities import ExperimentPermission, RegisteredModelPermission
from mlflow.server.kubeflow.db.models import SqlExperimentPermission, SqlRegisteredModelPermission
from mlflow.server.kubeflow.db import utils as dbutils
from mlflow.store.db.utils import create_sqlalchemy_engine_with_retry, _get_managed_session_maker
from mlflow.utils.uri import extract_db_type_from_uri


class SqlAlchemyStore:
    def init_db(self, db_uri):
        self.db_uri = db_uri
        self.db_type = extract_db_type_from_uri(db_uri)
        self.engine = create_sqlalchemy_engine_with_retry(db_uri)
        dbutils.migrate_if_needed(self.engine, "head")
        SessionMaker = sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = _get_managed_session_maker(SessionMaker, self.db_type)

    def create_experiment_permission(
        self, experiment_id: str, namespace: str
    ) -> ExperimentPermission:
        with self.ManagedSessionMaker() as session:
            try:
                perm = SqlExperimentPermission(experiment_id=experiment_id, namespace=namespace)
                session.add(perm)
                session.flush()
                return perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Experiment permission (experiment_id={experiment_id}, namespace={namespace}) "
                    f"already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                )

    def _get_experiment_permission(
        self, session, experiment_id: str, namespace: str
    ) -> SqlExperimentPermission:
        try:
            return (
                session.query(SqlExperimentPermission)
                .filter(
                    SqlExperimentPermission.experiment_id == experiment_id,
                    SqlExperimentPermission.namespace == namespace,
                )
                .one()
            )
        except NoResultFound:
            raise MlflowException(
                f"Experiment permission with experiment_id={experiment_id} and "
                f"namespace={namespace} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        except MultipleResultsFound:
            raise MlflowException(
                f"Found multiple experiment permissions with experiment_id={experiment_id} "
                f"and namespace={namespace}",
                INVALID_STATE,
            )

    def get_experiment_permission(self, experiment_id: str, namespace: str) -> ExperimentPermission:
        with self.ManagedSessionMaker() as session:
            return self._get_experiment_permission(
                session, experiment_id, namespace
            ).to_mlflow_entity()

    def list_experiment_permissions(self, namespace: str) -> List[ExperimentPermission]:
        with self.ManagedSessionMaker() as session:
            perms = (
                session.query(SqlExperimentPermission)
                .filter(SqlExperimentPermission.namespace == namespace)
                .all()
            )
            return [p.to_mlflow_entity() for p in perms]

    def delete_experiment_permission(self, experiment_id: str, namespace: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_experiment_permission(session, experiment_id, namespace)
            session.delete(perm)

    def create_registered_model_permission(
        self, name: str, namespace: str
    ) -> RegisteredModelPermission:
        with self.ManagedSessionMaker() as session:
            try:
                perm = SqlRegisteredModelPermission(name=name, namespace=namespace)
                session.add(perm)
                session.flush()
                return perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Registered model permission (name={name}, namespace={namespace}) "
                    f"already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                )

    def _get_registered_model_permission(
        self, session, name: str, namespace: str
    ) -> SqlRegisteredModelPermission:
        try:
            return (
                session.query(SqlRegisteredModelPermission)
                .filter(
                    SqlRegisteredModelPermission.name == name,
                    SqlRegisteredModelPermission.namespace == namespace,
                )
                .one()
            )
        except NoResultFound:
            raise MlflowException(
                f"Registered model permission with name={name} and namespace={namespace} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        except MultipleResultsFound:
            raise MlflowException(
                f"Found multiple registered model permissions with name={name} "
                f"and namespace={namespace}",
                INVALID_STATE,
            )

    def get_registered_model_permission(
        self, name: str, namespace: str
    ) -> RegisteredModelPermission:
        with self.ManagedSessionMaker() as session:
            return self._get_registered_model_permission(
                session, name, namespace
            ).to_mlflow_entity()

    def list_registered_model_permissions(self, namespace: str) -> List[RegisteredModelPermission]:
        with self.ManagedSessionMaker() as session:
            perms = (
                session.query(SqlRegisteredModelPermission)
                .filter(SqlRegisteredModelPermission.namespace == namespace)
                .all()
            )
            return [p.to_mlflow_entity() for p in perms]

    def delete_registered_model_permission(self, name: str, namespace: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_registered_model_permission(session, name, namespace)
            session.delete(perm)
