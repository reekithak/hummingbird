"""
Tests scikit-labebencoder converter.
"""
import unittest

import numpy as np
import torch
from skl2pytorch import convert_sklearn
from skl2pytorch.common.data_types import Int32TensorType
from sklearn.preprocessing import LabelEncoder


class TestSklearnLabelEncoderConverter(unittest.TestCase):
    def test_model_label_encoder(self):
        model = LabelEncoder()
        data = np.array([1, 4, 5, 2, 0, 2], dtype=np.int32)
        model.fit(data)

        pytorch_model = convert_sklearn(model, [("input", Int32TensorType([6, 1]))])
        self.assertTrue(pytorch_model is not None)
        self.assertTrue(np.allclose(model.transform(data), pytorch_model(torch.from_numpy(data)).data.numpy()))

    def test_model_label_encoder_str(self):
        model = LabelEncoder()
        # data = ['a', 'r', 'x', 'a']
        data = [
            "paris",
            "tokyo",
            "amsterdam",
            "tokyo",
        ]
        model.fit(data)

        # max word length is the smallest number which is divisible by 4 and larger than or equal to the length of any word
        max_word_length = 4
        num_columns = 4
        pytorch_model = convert_sklearn(model, [("input", Int32TensorType([4, 4, max_word_length // 4]))])

        ref = model.transform(data)
        pytorch_input = torch.from_numpy(np.array(data, dtype="|S" + str(max_word_length)).view(np.int32)).view(
            -1, num_columns, max_word_length // 4
        )
        mine = pytorch_model(pytorch_input).data.numpy()
        self.assertTrue(np.allclose(ref, mine))


if __name__ == "__main__":
    unittest.main()
