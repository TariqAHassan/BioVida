# coding: utf-8

"""

    Convert DICOM Data into a Python Dictionary
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Also see: https://github.com/darcymason/pydicom/issues/319.
import os
from biovida.support_tools.support_tools import cln, dicom


def _extract_numeric(value):
    """

    :param value:
    :return:
    """
    return float("".join((i for i in value if i.isdigit() or i == '.')))


def parse_age(value):
    """

    :param value:
    :return:
    """
    if not isinstance(value, str):
        raise TypeError('`value` must be a string.')
    elif len(value) > 4:
        return value

    if 'y' in value.lower():
        return _extract_numeric(value)
    elif 'm' in value.lower():
        return _extract_numeric(value) / 12.0
    else:
        return value


def parse_string_to_tuple(value):
    """

    :param value:
    :type value: ``str``
    :return:
    """
    braces = [['[', ']'], ['(', ')']]
    for (left, right) in braces:
        if left in value and right in value:
            value_split = value.replace(left, "").replace(right, "").split(",")
            value_split_cln = list(filter(None, map(cln, value_split)))
            if len(value_split_cln) == 0:
                return None
            try:
                to_return = tuple(map(_extract_numeric, value_split_cln))
            except:
                to_return = tuple(value_split_cln)
            return to_return[0] if len(to_return) == 1 else to_return
    else:
        raise ValueError("Cannot convert `value` to a tuple.")


def dicom_value_parse(key, value):
    """
    
    Try to convert ``value`` to a numeric or tuple of numerics.

    :param key:
    :param value:
    :return:
    """
    value = cln(str(value).replace("\'", "").replace("\"", ""))

    if not len(value) or value.lower() == 'none':
        return None

    if key.lower().endswith(' age') or key == 'PatientAge':
        try:
            return parse_age(value)
        except:
            return value
    else:
        try:
            return int(value)
        except:
            try:
                return float(value)
            except:
                try:
                    return parse_string_to_tuple(value)
                except:
                    return value


def dicom_object_dict_gen(dicom_object):
    """

    :param dicom_object:
    :type dicom_object: ``dicom.FileDataset``
    :return:
    """
    d = dict()
    for k in dicom_object.__dir__():
        if not k.startswith("__") and k != 'PixelData':
            try:
                value = dicom_object.data_element(k).value
                if type(value).__name__ != 'Sequence':
                    d[k] = dicom_value_parse(key=k, value=value)
            except:
                pass
    return d


def dicom_to_dict(dicom_file):
    """

    Convert the metadata associated with ``dicom_file`` into a python dictionary

    :param dicom_file: a path to a dicom file or the yield of ``dicom.read_file(FILE_PATH)``.
    :type dicom_file: ``FileDataset`` or ``str``
    :return: a dictionary with the dicom meta data.
    :rtype: ``dict``
    """
    if isinstance(dicom_file, str):
        if not os.path.isfile(dicom_file):
            raise FileNotFoundError("Could not locate '{0}'.".format(dicom_file))
        dicom_object = dicom.read_file(dicom_file)
    elif type(dicom_file).__name__ == 'FileDataset':
        dicom_object = dicom_file
    else:
        raise TypeError("`dicom_file` must be of type `dicom.FileDataset` or a string.")

    return dicom_object_dict_gen(dicom_object)
