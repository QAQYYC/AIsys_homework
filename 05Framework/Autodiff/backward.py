from dataclasses import dataclass

from typing import Callable

import numpy as np

dl_d = {}
tabe_list = []
_counter = 0

def fresh_name():
    global _counter
    _counter += 1
    return _counter

class Variable:
    def __init__(self, value, grad=None, name=None):
        self.value = value
        self.grad = grad
        self.name = name or fresh_name()

    @staticmethod
    def constant(value, name=None):
        val = Variable(value, grad=None, name=name)
        dl_d[val.name] = val
        print(f"v_{val.name} = Constant({value})")
        return val

    def __add__(self,other):
        return ops_add(self, other)

    def __sub__(self, other):
        return ops_sub(self, other)

    def __mul__(self, other):
        return ops_mul(self, other)

    def log(self):
        return ops_log(self)

    def sin(self):
        return ops_sin(self)

    def __repr__(self):
        return f"Variable(value={self.value}, grad={self.grad})"

@dataclass
class Tabe:
    input_names: list[str]
    output_names: list[str]
    operators: Callable[..., list[float]]


def ops_add(self, other):
    print("Adding operation")
    out = Variable.constant(self.value + other.value)

    def aggregation(output):
        return [output.grad, output.grad]

    op = Tabe(input_names=[self.name, other.name], output_names=[out.name], operators=aggregation)
    tabe_list.append(op)
    dl_d[out.name] = out
    return out

def ops_mul(self, other):
    print("Multiplying operation")
    out = Variable.constant(self.value * other.value)

    def aggregation(output):
        return [other.value * output.grad, self.value * output.grad]

    op = Tabe(input_names=[self.name, other.name], output_names=[out.name], operators=aggregation)
    tabe_list.append(op)
    dl_d[out.name] = out
    return out

def ops_sub(self, other):
    print("Subtracting operation")
    out = Variable.constant(self.value - other.value)

    def aggregation(output):
        return [output.grad, -output.grad]

    op = Tabe(input_names=[self.name, other.name], output_names=[out.name], operators=aggregation)
    tabe_list.append(op)
    dl_d[out.name] = out
    return out

def ops_log(self):
    print("Logarithm operation")
    out = Variable.constant(np.log(self.value))
    def aggregation(output):
        return [(1 / self.value) * output.grad]

    op = Tabe(input_names=[self.name], output_names=[out.name], operators=aggregation)
    tabe_list.append(op)
    dl_d[out.name] = out
    return out

def ops_sin(self):
    print("Sine operation")
    out = Variable.constant(np.sin(self.value))
    def aggregation(output):
        return [np.cos(self.value) * output.grad]

    op = Tabe(input_names=[self.name], output_names=[out.name], operators=aggregation)
    tabe_list.append(op)
    dl_d[out.name] = out
    return out

def backward():
    for tabe in reversed(tabe_list):
        output_vars = [dl_d[name] for name in tabe.output_names]
        backward_grad = tabe.operators(*output_vars)
        for grad, name in zip(backward_grad, tabe.input_names):
            if dl_d[name].grad is None:
                dl_d[name].grad = grad
            else:
                dl_d[name].grad += grad

if __name__ == '__main__':
    x1 = Variable.constant(value=2.0, name='-1')
    x2 = Variable.constant(value=5.0, name='0')
    z = x1.log() + x1 * x2 - x2.sin()
    z.grad = 1.0
    backward()
    print(x1, x2)










