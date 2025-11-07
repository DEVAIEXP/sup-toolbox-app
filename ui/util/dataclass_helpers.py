# Copyright 2025 The DEVAIEXP Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import functools
from dataclasses import Field
from typing import Any, Dict, List, Tuple, Type

from typing_extensions import Literal


# General Helper Functions
def get_nested_attr(obj: Any, path: str, default: Any = None) -> Any:
    """
    Accesses a nested attribute from an object using 'dot notation'.

    Returns a default value if any attribute in the chain is missing or if
    the object itself is None.

    Args:
        obj:
            The root object.
        path:
            A string path like "parent.child.attribute".
        default:
            The value to return if the path is not found.

    Returns:
        The value of the nested attribute or the default value.
    """
    try:
        return functools.reduce(getattr, path.split("."), obj)
    except AttributeError:
        return default


def _set_nested_dict_value(d: dict, path: str, value: Any):
    """
    Sets a value in a nested dictionary using 'dot notation'.

    Args:
        d:
            The dictionary to modify.
        path:
            A string path like "parent.child.key".
        value:
            The new value to set.
    """
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def get_nested_field(dc_type: Type, path: str) -> Field:
    """
    Retrieves the dataclasses.Field object for a nested attribute.

    Args:
        dc_type:
            The top-level dataclass type.
        path:
            A string path like "parent.child.field_name".

    Returns:
        The dataclasses.Field definition object.
    """
    parts = path.split(".")
    current_type = dc_type
    for i, part in enumerate(parts):
        field_info = current_type.__dataclass_fields__[part]
        if i == len(parts) - 1:
            return field_info
        current_type = field_info.type
    raise KeyError(f"Field path '{path}' could not be fully resolved in {dc_type}.")


def get_nested_parent_and_field(obj: object, path: str) -> Tuple[object, str]:
    """
    Navigates a nested object path and returns the parent object and the
    final field name.

    Example: for path "general.sampler.seed", returns (sampler_object, "seed").

    Args:
        obj:
            The root object to navigate.
        path:
            A dot-separated string path.

    Returns:
        A tuple containing the direct parent object and the final field name.
    """
    parts = path.split(".")
    parent = obj
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


# Dataclass Instantiation Helper
def dataclass_from_dict(data_class: Type, data: dict) -> Any:
    """
    Recursively constructs a dataclass instance from a dictionary.

    This helper is essential for creating a dataclass instance from dictionary
    data, especially when dealing with nested dataclasses. It correctly
    handles nested structures by recursively calling itself.

    Args:
        data_class:
            The dataclass type to instantiate (e.g., `AppSettings`).
        data:
            A dictionary containing the data to populate the instance.
            Keys should correspond to field names in the dataclass.

    Returns:
        An instance of `data_class` populated with data from the dictionary.
    """
    field_names = {f.name for f in dataclasses.fields(data_class)}
    filtered_data = {k: v for k, v in data.items() if k in field_names}

    for f in dataclasses.fields(data_class):
        field_name = f.name
        field_type = f.type

        if field_name in filtered_data and dataclasses.is_dataclass(field_type):
            if isinstance(filtered_data[field_name], dict):
                nested_instance = dataclass_from_dict(field_type, filtered_data[field_name])
                filtered_data[field_name] = nested_instance

    return data_class(**filtered_data)


# The Core Engine for Dynamic Dataclass Reconstruction
def _rebuild_dataclass_recursively(dc_type: Type, path: str, new_field: Field) -> Type:
    """
    Internal function to recursively rebuild a nested dataclass structure.

    It replaces an entire target field definition, allowing for changes to
    any part of the field (type, metadata, default value, etc.).

    Args:
        dc_type:
            The current dataclass type to rebuild.
        path:
            The remaining dot-separated path to the target field.
        new_field:
            The complete new dataclasses.Field object to insert.

    Returns:
        A new, dynamically created dataclass type with the updated structure.
    """
    parts = path.split(".")
    field_to_change = parts[0]

    if len(parts) == 1:
        new_fields_spec = []
        for f in dataclasses.fields(dc_type):
            if f.name == field_to_change:
                new_spec = (f.name, new_field.type, new_field)
                new_fields_spec.append(new_spec)
            else:
                new_fields_spec.append((f.name, f.type, f))
        class_name = f"Updated_{dc_type.__name__.rpartition('_')[-1]}"
        return dataclasses.make_dataclass(
            class_name,
            new_fields_spec,
            bases=dc_type.__bases__,
        )
    else:
        nested_dc_type = dc_type.__dataclass_fields__[field_to_change].type
        remaining_path = ".".join(parts[1:])
        rebuilt_nested_type = _rebuild_dataclass_recursively(nested_dc_type, remaining_path, new_field)

        new_parent_fields_spec = []
        for f in dataclasses.fields(dc_type):
            if f.name == field_to_change:
                new_parent_nested_field = dataclasses.field(default_factory=f.default_factory, metadata=f.metadata, init=f.init)
                new_spec = (f.name, rebuilt_nested_type, new_parent_nested_field)
                new_parent_fields_spec.append(new_spec)
            else:
                new_parent_fields_spec.append((f.name, f.type, f))
        class_name = f"Updated_{dc_type.__name__.rpartition('_')[-1]}"
        return dataclasses.make_dataclass(class_name, new_parent_fields_spec, bases=dc_type.__bases__)


def apply_dynamic_changes(dc_instance: object, rules: dict) -> object:
    """
    Processes a dataclass instance against a set of UI rules.

    This version ensures consistency by always rebuilding the dataclass
    instance if any rule that modifies the class structure (like changing
    options or visibility) is present. This prevents state inconsistencies
    within the Gradio UI lifecycle.

    Args:
        dc_instance:
            The source dataclass instance.
        rules:
            A dictionary containing the UI rules.

    Returns:
        A new dataclass instance of a dynamically created type, with all
        structural rules applied.
    """
    current_dc = dc_instance
    actions_to_process: List[Dict] = []

    # Part 1: Gather dynamic actions based on current field values
    if "dynamic_dependencies" in rules:
        for modifier_path, rule_group in rules["dynamic_dependencies"].items():
            modifier_value = get_nested_attr(current_dc, modifier_path)
            for action in rule_group.get("actions", []):
                mapping = action.get("mapping", {})
                if modifier_value in mapping:
                    outcome = mapping[modifier_value]
                    actions_to_process.append({**action, "outcome": outcome})

    # Part 2: Gather unconditional "on_load" actions
    if "on_load_actions" in rules:
        actions_to_process.extend(rules["on_load_actions"])

    # Part 3: Process all gathered actions
    for action in actions_to_process:
        action_type = action.get("type")
        target_path = action.get("target_field_path")
        outcome = action.get("outcome")  # Will be None for on_load_actions

        if not all([action_type, target_path]):
            continue

        target_field_info = get_nested_field(type(current_dc), target_path)

        # Handle actions that require rebuilding the dataclass
        if action_type in ["update_options", "update_visibility"]:
            new_field_type = target_field_info.type
            new_metadata = dict(target_field_info.metadata)

            if action_type == "update_options":
                new_options = outcome.get("options", [])
                new_field_type = Literal[tuple(new_options)] if new_options else str

            elif action_type == "update_visibility":
                is_visible = action.get("visible")
                if is_visible is not None:
                    new_metadata["visible"] = is_visible

            # Create the new field definition based on the changes
            new_field = dataclasses.field(
                default=target_field_info.default,
                default_factory=target_field_info.default_factory,
                init=target_field_info.init,
                repr=target_field_info.repr,
                hash=target_field_info.hash,
                compare=target_field_info.compare,
                metadata=new_metadata,
            )
            new_field.type = new_field_type
            new_field.name = target_field_info.name

            # Rebuild the dataclass and replace the current instance
            NewDcClass = _rebuild_dataclass_recursively(type(current_dc), target_path, new_field)
            values_dict = dataclasses.asdict(current_dc)

            # Correct the value if it became invalid after an `update_options` action
            if action_type == "update_options":
                current_value = get_nested_attr(current_dc, target_path)
                new_options = outcome.get("options", [])
                new_default = outcome.get("new_default")
                should_update = current_value not in new_options or (new_default is not None and current_value != new_default)
                if should_update:
                    value_to_set = new_default if new_default is not None else (new_options[0] if new_options else None)
                    if value_to_set is not None:
                        _set_nested_dict_value(values_dict, target_path, value_to_set)

            # The `current_dc` instance is replaced. The next loop iteration
            # will operate on this newly created instance.
            current_dc = dataclass_from_dict(NewDcClass, values_dict)

        # Handle actions that modify the instance directly
        elif action_type == "set_value":
            new_value = outcome
            current_value = get_nested_attr(current_dc, target_path)
            if new_value != current_value:
                parent_obj, field_name = get_nested_parent_and_field(current_dc, target_path)
                setattr(parent_obj, field_name, new_value)

    return current_dc
