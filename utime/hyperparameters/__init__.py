import logging
from utime import Defaults
from psg_utils.utils import ensure_list_or_tuple
from yamlhparams import YAMLHParams as _YAMLHParams

logger = logging.getLogger(__name__)


def _handle_channel_sampling_group_renaming(hparams):
    """
    Replace 'access_time_channel_sampling_groups' and/or 'load_time_channel_sampling_groups'
    attributes with the new common attribute 'channel_sampling_groups'.
    Required as per utime v1.0.0 due to API changes in psg_utils package.
    """
    new_name = "channel_sampling_groups"
    for deprecated_name in ('access_time_channel_sampling_groups', 'load_time_channel_sampling_groups'):
        value = hparams.get(deprecated_name, None)
    if value is not None:
        logger.warning(f"Found deprecated hyperparameter value '{deprecated_name}' in hyperparameter file "
                       f"at path {hparams.yaml_path}. Renaming to '{new_name}' and saving "
                       f"hyperparameters to disk.")
        hparams.delete_group(deprecated_name)
        hparams.set_group(f"/{new_name}", value=value)
        hparams.save_current()


def _handle_metrics_renaming(hparams):
    """
    Replace metrics specified in the old TF function 'sparse_metric_name' format with
    new class 'SparseMetricName' format. Required as per utime v1.0.0
    """
    map_ = {
        "sparse_categorical_accuracy": "SparseCategoricalAccuracy",
        "sparse_categorical_crossentropy": "SparseCategoricalCrossentropy",
        "sparse_top_k_categorical_accuracy": "SparseTopKCategoricalAccuracy"
    }
    try:
        metrics = hparams.get_group('/fit/metrics')
    except KeyError:
        pass
    else:
        any_replaced = False
        for i, metric in enumerate(ensure_list_or_tuple(metrics or [])):
            if metric in map_:
                logger.warning(f"Found deprecated metrics naming of '{metric}' in hyperparameter file "
                               f"at path {hparams.yaml_path}. Renaming to '{map_[metric]}' and "
                               f"saving hyperparameters to disk.")
                any_replaced = True
                metrics[i] = map_[metric]
        if any_replaced:
            hparams.set_group("/fit/metrics", value=metrics, overwrite=True)
            hparams.save_current()


def _handle_version_format_changes(hparams):
    """
    Check if hparams file has old format __VERSION__, __COMMIT__ and/or __BRANCH__ tags.
    If so, warn the user and suggest them to remove them manually (we do not delete as some users
    may need to manually refer to these versions as no version control will be made against them anymore).
    """
    tags_to_check = ('__VERSION__', '__COMMIT__', '__BRANCH__')
    for tag in tags_to_check:
        if hparams.get(tag):
            logger.warning(f"Found deprecated version controlling attribute naming of '{tag}' in "
                           f"hyperparameter file at path {hparams.yaml_path}. "
                           f"Since utime v1.0.0 this and other version control related attributes have been "
                           f"renamed. The new attributes will be automatically added to this file and the old tags "
                           f"will remain. However, no version controlling will be made against the OLD tags. "
                           f"To suppress this warning, delete the old tag manually from the hyperparameters file.")


def _handle_period_length_sec(hparams):
    any_changes = False
    for data_group in ("train_data", "val_data", "test_data"):
        try:
            group = hparams.get_group(data_group)
            period_length_sec = group.get('period_length_sec')
            if period_length_sec:
                logger.warning(f"Found deprecated 'period_length_sec' parameter in data group '{data_group}' "
                               f"in hyperparameter file at path {hparams.yaml_path}. Since utime v1.1.0 this "
                               f"parameter has been renamed to 'period_length' with units specified by the new "
                               f"'time_unit' parameter. Setting 'period_length' and time_unit=SECONDS.")
                hparams.delete_group(f"{data_group}/period_length_sec")
                hparams.set_group(f"{data_group}/period_length", value=period_length_sec)
                hparams.set_group(f"{data_group}/time_unit", value="SECONDS")
                any_changes = True
        except KeyError:
            pass
    if any_changes:
        hparams.save_current()


def _handle_strip_func_str_renaming(hparams):
    if hparams.get('strip_func') and hparams['strip_func'].get('strip_func_str'):
        logger.warning("Renaming parameter 'strip_func_str' -> 'strip_func'")
        mem = hparams['strip_func']['strip_func_str']
        del hparams['strip_func']['strip_func_str']
        hparams['strip_func']['strip_func'] = mem
        hparams.save_current()


def _handle_weights_file_name(hparams):
    if hparams.get('weights_file_name'):
        logger.warning("Removing parameter 'weights_file_name' from hparams object (DEPRECATED)")
        del hparams['weights_file_name']
        hparams.save_current()


def check_deprecated_params(hparams):
    _handle_channel_sampling_group_renaming(hparams)
    _handle_metrics_renaming(hparams)
    _handle_version_format_changes(hparams)
    _handle_period_length_sec(hparams)
    _handle_strip_func_str_renaming(hparams)
    _handle_weights_file_name(hparams)


class YAMLHParams(_YAMLHParams):
    """
    Wrapper around the yamlhparams.YAMLHParams object to pass the utime package name for VC.
    Also allows to disable VC with no_version_control parameter.
    """
    def __init__(self, yaml_path, no_version_control=False):
        vc = Defaults.PACKAGE_NAME if not no_version_control else None
        super(YAMLHParams, self).__init__(yaml_path,
                                          version_control_package_name=vc,
                                          check_deprecated_params_func=check_deprecated_params)
