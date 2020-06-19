# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import torch
import numpy as np
from ..common._registration import register_converter


class StringLabelEncoder(torch.nn.Module):
    def __init__(self, classes, device):
        super(StringLabelEncoder, self).__init__()
        self.regression = False
        self.num_columns = len(classes)
        self.max_word_length = max([max([len(c) for c in cat]) for cat in classes])
        while self.max_word_length % 4 != 0:
            self.max_word_length += 1

        data_type = "|S" + str(self.max_word_length)

        # sort the classes and convert to torch.int32
        classes_conv = torch.from_numpy(np.array(sorted(set(classes)), dtype=data_type).view(np.int32))

        self.condition_tensors = torch.nn.Parameter(torch.IntTensor(classes_conv), requires_grad=False)

    def forward(self, x):
        x = x.view(-1, 1)
        try:
            return torch.from_numpy(np.array([(self.condition_tensors == v).nonzero() for v in x]))
        except KeyError:
            raise ValueError(
                "x ({}) contains previously unseen labels. condition_tensors: {}".format(x, self.condition_tensors)
            )


class NumericLabelEncoder(torch.nn.Module):
    def __init__(self, classes, device):
        super(NumericLabelEncoder, self).__init__()
        self.regression = False
        self.check_tensor = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)

    def forward(self, x):
        x = x.view(-1, 1)
        try:  # has GPU.
            return torch.argmax(torch.eq(x, self.check_tensor), dim=1)
        except Exception:  # Tensorflow issue?: https://github.com/allenai/allennlp/issues/3455
            # fix by casting bool to int
            # TODO: clearly there is a function that does this for us....
            def convert(row):
                return [0 if not x else 1 for x in row]

            tmp = torch.tensor([convert(col) for col in torch.eq(x, self.check_tensor)])
            return torch.argmax(tmp, dim=1)


def convert_sklearn_label_encoder(operator, device, extra_config):
    # TODO Add docstring here!  Please see example at
    # https://github.com/microsoft/hummingbird/blob/master/hummingbird/ml/operator_converters/skl_linear.py#L57
    if all([type(x) == str for x in operator.raw_operator.classes_]):
        raise RuntimeError(
            "Hummingbird currently supports only integer labels for class labels. Please file an issue at https://github.com/microsoft/hummingbird."
        )
    elif all([type(x) in [int, np.int32, np.int64] for x in operator.raw_operator.classes_]):
        return NumericLabelEncoder(operator.raw_operator.classes_, device)
    else:
        return StringLabelEncoder(operator.raw_operator.classes_, device)


register_converter("SklearnLabelEncoder", convert_sklearn_label_encoder)
