"""Implementing a custom Tensorflow Op in C++ has some advantages and disadvantages

Advantages:
- GPU support without any hacks
- Access to raw tensors without conversion to numpy
- Custom Ops will be serializable

Disadavantages:
- C++ Code has to be build with correct parameters and ABI for present Tensorflow version (best integrated into Tensorflow build)

"""


# raise NotImplementedError()
