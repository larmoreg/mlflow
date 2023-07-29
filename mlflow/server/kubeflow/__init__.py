"""
Usage
-----

.. code-block:: bash

    mlflow server --app-name kubeflow-auth
"""

import logging
import os
import re
import requests
from typing import Callable, Dict, Optional, Any

from flask import Flask, request, make_response, Response

from mlflow import MlflowException
from mlflow.entities import Experiment
from mlflow.entities.model_registry import RegisteredModel
from mlflow.server import app
from mlflow.server.kubeflow.config import read_auth_config
from mlflow.server.kubeflow.permissions import get_permission, Permission, READ, EDIT
from mlflow.server.kubeflow.sqlalchemy_store import SqlAlchemyStore
from mlflow.server.handlers import (
    _get_request_message,
    _get_tracking_store,
    _get_model_registry_store,
    catch_mlflow_exception,
    get_endpoints,
)
from mlflow.store.entities import PagedList
from mlflow.protos.databricks_pb2 import (
    ErrorCode,
    BAD_REQUEST,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.protos.service_pb2 import (
    GetExperiment,
    GetRun,
    ListArtifacts,
    GetMetricHistory,
    CreateRun,
    UpdateRun,
    LogMetric,
    LogParam,
    SetTag,
    DeleteExperiment,
    RestoreExperiment,
    RestoreRun,
    DeleteRun,
    UpdateExperiment,
    LogBatch,
    DeleteTag,
    SetExperimentTag,
    GetExperimentByName,
    LogModel,
    CreateExperiment,
    SearchExperiments,
)
from mlflow.protos.model_registry_pb2 import (
    GetRegisteredModel,
    DeleteRegisteredModel,
    UpdateRegisteredModel,
    RenameRegisteredModel,
    GetLatestVersions,
    CreateModelVersion,
    GetModelVersion,
    DeleteModelVersion,
    UpdateModelVersion,
    TransitionModelVersionStage,
    GetModelVersionDownloadUri,
    SetRegisteredModelTag,
    DeleteRegisteredModelTag,
    SetModelVersionTag,
    DeleteModelVersionTag,
    SetRegisteredModelAlias,
    DeleteRegisteredModelAlias,
    GetModelVersionByAlias,
    CreateRegisteredModel,
    SearchRegisteredModels,
)
from mlflow.utils.proto_json_utils import parse_dict, message_to_json
from mlflow.utils.search_utils import SearchUtils
from mlflow.utils.rest_utils import _REST_API_PATH_PREFIX

KFAM_URL = os.getenv("KFAM_URL", "http://profiles-kfam.kubeflow.svc.cluster.local:8081/kfam")
NAMESPACE_COOKIE = os.getenv("NAMESPACE_COOKIE", "kubeflow_namespace")
USERID_HEADER = os.getenv("USERID_HEADER", "kubeflow-userid")

_logger = logging.getLogger(__name__)

auth_config = read_auth_config()
store = SqlAlchemyStore()


def is_unprotected_route(path: str) -> bool:
    return path.startswith(("/static", "/favicon.ico", "/health"))


def make_kubeflow_auth_response() -> Response:
    res = make_response(
        "You are not authenticated. Please see "
        "https://www.mlflow.org/docs/latest/auth/index.html#authenticating-to-mlflow "
        "on how to authenticate."
    )
    res.status_code = 401
    res.headers["WWW-Authenticate"] = 'Basic realm="mlflow"'
    return res


def make_forbidden_response() -> Response:
    res = make_response("Permission denied")
    res.status_code = 403
    return res


def _get_request_param(param: str) -> str:
    if request.method == "GET":
        args = request.args
    elif request.method in ("POST", "PATCH", "DELETE"):
        args = request.json
    else:
        raise MlflowException(
            f"Unsupported HTTP method '{request.method}'",
            BAD_REQUEST,
        )

    if param not in args:
        # Special handling for run_id
        if param == "run_id":
            return _get_request_param("run_uuid")
        raise MlflowException(
            f"Missing value for required parameter '{param}'. "
            "See the API docs for more information about request parameters.",
            INVALID_PARAMETER_VALUE,
        )
    return args[param]


def _get_permission_from_kfam(namespace: str) -> Permission:
    username = _get_username()
    response = requests.get(
        f"{KFAM_URL}/v1/bindings?namespace={namespace}&username={username}",
        cookies=request.cookies,
    )
    response.raise_for_status()
    permissions = response.json()

    if bindings := permissions.get("bindings"):
        roles = [bindings[0]["RoleRef"]["name"] for binding in bindings]
        if any(role == "admin" or role == "edit" for role in roles):
            return EDIT
        else:
            return READ
    return get_permission(auth_config.default_permission)


def _get_permission_from_kfam_or_default(
    store_permission_func: Callable[[], Any], kfam_permission_func: Callable[[], Permission]
) -> Permission:
    """
    Attempts to get permission from store,
    and returns default permission if no record is found.
    """
    try:
        store_permission_func()
        return kfam_permission_func()
    except MlflowException as e:
        if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return get_permission(auth_config.default_permission)
        else:
            raise


def _get_permission_from_experiment_id() -> Permission:
    experiment_id = _get_request_param("experiment_id")
    namespace = _get_namespace()
    return _get_permission_from_kfam_or_default(
        lambda: store.get_experiment_permission(experiment_id, namespace),
        lambda: _get_permission_from_kfam(namespace),
    )


_EXPERIMENT_ID_PATTERN = re.compile(r"^(\d+)/")


def _get_experiment_id_from_view_args():
    if artifact_path := request.view_args.get("artifact_path"):
        if m := _EXPERIMENT_ID_PATTERN.match(artifact_path):
            return m.group(1)
    return None


def _get_permission_from_experiment_id_artifact_proxy() -> Permission:
    if experiment_id := _get_experiment_id_from_view_args():
        namespace = _get_namespace()
        return _get_permission_from_kfam_or_default(
            lambda: store.get_experiment_permission(experiment_id, namespace),
            lambda: _get_permission_from_kfam(namespace),
        )
    return get_permission(auth_config.default_permission)


def _get_permission_from_experiment_name() -> Permission:
    experiment_name = _get_request_param("experiment_name")
    store_exp = _get_tracking_store().get_experiment_by_name(experiment_name)
    if store_exp is None:
        raise MlflowException(
            f"Could not find experiment with name {experiment_name}",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )
    namespace = _get_namespace()
    return _get_permission_from_kfam_or_default(
        lambda: store.get_experiment_permission(store_exp.experiment_id, namespace),
        lambda: _get_permission_from_kfam(namespace),
    )


def _get_permission_from_run_id() -> Permission:
    # run permissions inherit from parent resource (experiment)
    # so we just get the experiment permission
    run_id = _get_request_param("run_id")
    run = _get_tracking_store().get_run(run_id)
    experiment_id = run.info.experiment_id
    namespace = _get_namespace()
    return _get_permission_from_kfam_or_default(
        lambda: store.get_experiment_permission(experiment_id, namespace),
        lambda: _get_permission_from_kfam(namespace),
    )


def _get_permission_from_registered_model_name() -> Permission:
    name = _get_request_param("name")
    namespace = _get_namespace()
    return _get_permission_from_kfam_or_default(
        lambda: store.get_registered_model_permission(name, namespace),
        lambda: _get_permission_from_kfam(namespace),
    )


def validate_can_read_experiment():
    return _get_permission_from_experiment_id().can_read


def validate_can_read_experiment_by_name():
    return _get_permission_from_experiment_name().can_read


def validate_can_update_experiment():
    return _get_permission_from_experiment_id().can_update


def validate_can_delete_experiment():
    return _get_permission_from_experiment_id().can_delete


def validate_can_read_experiment_artifact_proxy():
    return _get_permission_from_experiment_id_artifact_proxy().can_read


def validate_can_update_experiment_artifact_proxy():
    return _get_permission_from_experiment_id_artifact_proxy().can_update


def validate_can_read_run():
    return _get_permission_from_run_id().can_read


def validate_can_update_run():
    return _get_permission_from_run_id().can_update


def validate_can_delete_run():
    return _get_permission_from_run_id().can_delete


def validate_can_read_registered_model():
    return _get_permission_from_registered_model_name().can_read


def validate_can_update_registered_model():
    return _get_permission_from_registered_model_name().can_update


def validate_can_delete_registered_model():
    return _get_permission_from_registered_model_name().can_delete


BEFORE_REQUEST_HANDLERS = {
    # Routes for experiments
    GetExperiment: validate_can_read_experiment,
    GetExperimentByName: validate_can_read_experiment_by_name,
    DeleteExperiment: validate_can_delete_experiment,
    RestoreExperiment: validate_can_delete_experiment,
    UpdateExperiment: validate_can_update_experiment,
    SetExperimentTag: validate_can_update_experiment,
    # Routes for runs
    CreateRun: validate_can_update_experiment,
    GetRun: validate_can_read_run,
    DeleteRun: validate_can_delete_run,
    RestoreRun: validate_can_delete_run,
    UpdateRun: validate_can_update_run,
    LogMetric: validate_can_update_run,
    LogBatch: validate_can_update_run,
    LogModel: validate_can_update_run,
    SetTag: validate_can_update_run,
    DeleteTag: validate_can_update_run,
    LogParam: validate_can_update_run,
    GetMetricHistory: validate_can_read_run,
    ListArtifacts: validate_can_read_run,
    # Routes for model registry
    GetRegisteredModel: validate_can_read_registered_model,
    DeleteRegisteredModel: validate_can_delete_registered_model,
    UpdateRegisteredModel: validate_can_update_registered_model,
    RenameRegisteredModel: validate_can_update_registered_model,
    GetLatestVersions: validate_can_read_registered_model,
    CreateModelVersion: validate_can_update_registered_model,
    GetModelVersion: validate_can_read_registered_model,
    DeleteModelVersion: validate_can_delete_registered_model,
    UpdateModelVersion: validate_can_update_registered_model,
    TransitionModelVersionStage: validate_can_update_registered_model,
    GetModelVersionDownloadUri: validate_can_read_registered_model,
    SetRegisteredModelTag: validate_can_update_registered_model,
    DeleteRegisteredModelTag: validate_can_update_registered_model,
    SetModelVersionTag: validate_can_update_registered_model,
    DeleteModelVersionTag: validate_can_delete_registered_model,
    SetRegisteredModelAlias: validate_can_update_registered_model,
    DeleteRegisteredModelAlias: validate_can_delete_registered_model,
    GetModelVersionByAlias: validate_can_read_registered_model,
}


def get_before_request_handler(request_class):
    return BEFORE_REQUEST_HANDLERS.get(request_class)


BEFORE_REQUEST_VALIDATORS = {
    (http_path, method): handler
    for http_path, handler, methods in get_endpoints(get_before_request_handler)
    for method in methods
}


def _is_proxy_artifact_path(path: str) -> bool:
    return path.startswith(f"{_REST_API_PATH_PREFIX}/mlflow-artifacts/artifacts/")


def _get_proxy_artifact_validator(
    method: str, view_args: Optional[Dict[str, Any]]
) -> Optional[Callable[[], bool]]:
    if view_args is None:
        return validate_can_read_experiment_artifact_proxy  # List

    return {
        "GET": validate_can_read_experiment_artifact_proxy,  # Download
        "PUT": validate_can_update_experiment_artifact_proxy,  # Upload
        "DELETE": validate_can_update_experiment_artifact_proxy,  # Delete
    }.get(method)


def _get_namespace() -> str:
    return request.cookies.get(NAMESPACE_COOKIE)


def _get_username() -> str:
    return request.headers.get(USERID_HEADER)


@catch_mlflow_exception
def _before_request():
    if is_unprotected_route(request.path):
        return

    if not _get_namespace() or not _get_username():
        return make_kubeflow_auth_response()

    # authorization
    if validator := BEFORE_REQUEST_VALIDATORS.get((request.path, request.method)):
        if not validator():
            return make_forbidden_response()
    elif _is_proxy_artifact_path(request.path):
        if validator := _get_proxy_artifact_validator(request.method, request.view_args):
            if not validator():
                return make_forbidden_response()


def create_experiment_permission(resp: Response):
    response_message = CreateExperiment.Response()
    parse_dict(resp.json, response_message)
    experiment_id = response_message.experiment_id
    namespace = _get_namespace()
    store.create_experiment_permission(experiment_id, namespace)


def create_registered_model_permission(resp: Response):
    response_message = CreateRegisteredModel.Response()
    parse_dict(resp.json, response_message)
    name = response_message.registered_model.name
    namespace = _get_namespace()
    store.create_registered_model_permission(name, namespace)


def delete_experiment_permission(resp: Response):
    response_message = CreateExperiment.Response()
    parse_dict(resp.json, response_message)
    experiment_id = response_message.experiment_id
    namespace = _get_namespace()
    store.delete_experiment_permission(experiment_id, namespace)


def delete_registered_model_permission(resp: Response):
    response_message = CreateRegisteredModel.Response()
    parse_dict(resp.json, response_message)
    name = response_message.registered_model.name
    namespace = _get_namespace()
    store.detele_registered_model_permission(name, namespace)


def filter_search_experiments(resp: Response):
    response_message = SearchExperiments.Response()
    parse_dict(resp.json, response_message)

    # fetch permissions
    namespace = _get_namespace()
    perms = store.list_experiment_permissions(namespace)
    default_can_read = _get_permission_from_kfam(namespace).can_read
    can_read = [e.experiment_id for e in perms if default_can_read]

    # filter out unreadable
    for e in list(response_message.experiments):
        if e.experiment_id not in can_read:
            response_message.experiments.remove(e)

    # re-fetch to fill max results
    request_message = _get_request_message(SearchExperiments())
    while (
        len(response_message.experiments) < request_message.max_results
        and response_message.next_page_token != ""
    ):
        refetched: PagedList[Experiment] = _get_tracking_store().search_experiments(
            view_type=request_message.view_type,
            max_results=request_message.max_results,
            order_by=request_message.order_by,
            filter_string=request_message.filter,
            page_token=response_message.next_page_token,
        )
        refetched = refetched[: request_message.max_results - len(response_message.experiments)]
        if len(refetched) == 0:
            response_message.next_page_token = ""
            break

        refetched_readable_proto = [e.to_proto() for e in refetched if e.experiment_id in can_read]
        response_message.experiments.extend(refetched_readable_proto)

        # recalculate next page token
        start_offset = SearchUtils.parse_start_offset_from_page_token(
            response_message.next_page_token
        )
        final_offset = start_offset + len(refetched)
        response_message.next_page_token = SearchUtils.create_page_token(final_offset)

    resp.data = message_to_json(response_message)


def filter_search_registered_models(resp: Response):
    response_message = SearchRegisteredModels.Response()
    parse_dict(resp.json, response_message)

    # fetch permissions
    namespace = _get_namespace()
    perms = store.list_registered_model_permissions(namespace)
    default_can_read = _get_permission_from_kfam(namespace).can_read
    can_read = [rm.name for rm in perms if default_can_read]

    # filter out unreadable
    for rm in list(response_message.registered_models):
        if rm.name not in can_read:
            response_message.registered_models.remove(rm)

    # re-fetch to fill max results
    request_message = _get_request_message(SearchRegisteredModels())
    while (
        len(response_message.registered_models) < request_message.max_results
        and response_message.next_page_token != ""
    ):
        refetched: PagedList[
            RegisteredModel
        ] = _get_model_registry_store().search_registered_models(
            filter_string=request_message.filter,
            max_results=request_message.max_results,
            order_by=request_message.order_by,
            page_token=response_message.next_page_token,
        )
        refetched = refetched[
            : request_message.max_results - len(response_message.registered_models)
        ]
        if len(refetched) == 0:
            response_message.next_page_token = ""
            break

        refetched_readable_proto = [rm.to_proto() for rm in refetched if rm.name in can_read]
        response_message.registered_models.extend(refetched_readable_proto)

        # recalculate next page token
        start_offset = SearchUtils.parse_start_offset_from_page_token(
            response_message.next_page_token
        )
        final_offset = start_offset + len(refetched)
        response_message.next_page_token = SearchUtils.create_page_token(final_offset)

    resp.data = message_to_json(response_message)


AFTER_REQUEST_PATH_HANDLERS = {
    CreateExperiment: create_experiment_permission,
    CreateRegisteredModel: create_registered_model_permission,
    DeleteExperiment: delete_experiment_permission,
    DeleteRegisteredModel: delete_registered_model_permission,
    SearchExperiments: filter_search_experiments,
    SearchRegisteredModels: filter_search_registered_models,
}


def get_after_request_handler(request_class):
    return AFTER_REQUEST_PATH_HANDLERS.get(request_class)


AFTER_REQUEST_HANDLERS = {
    (http_path, method): handler
    for http_path, handler, methods in get_endpoints(get_after_request_handler)
    for method in methods
}


@catch_mlflow_exception
def _after_request(resp: Response):
    if 400 <= resp.status_code < 600:
        return resp

    if handler := AFTER_REQUEST_HANDLERS.get((request.path, request.method)):
        handler(resp)
    return resp


def create_app(app: Flask = app):
    """
    A factory to enable authentication and authorization for the MLflow server.

    :param app: The Flask app to enable authentication and authorization for.
    :return: The app with authentication and authorization enabled.
    """
    store.init_db(auth_config.database_uri)

    app.before_request(_before_request)
    app.after_request(_after_request)

    return app
