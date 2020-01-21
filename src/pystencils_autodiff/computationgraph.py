#
# Distributed under terms of the GPLv3 license.

"""

"""

from itertools import chain

import pystencils
from pystencils import Field
from pystencils.astnodes import KernelFunction
from pystencils.kernel_wrapper import KernelWrapper
from pystencils_autodiff.graph_datahandling import KernelCall, Swap, TimeloopRun


class ComputationGraph:
    class FieldWriteCounter:
        def __init__(self, field, counter=0):
            self.field = field
            self.counter = counter

        def __hash__(self):
            return hash((self.field, self.counter))

        def next(self):
            return self.__class__(self.field, self.counter + 1)

        @property
        def name(self):
            return self.field.name

        def __str__(self):
            return self.__repr__()

        def __repr__(self):
            return f'{self.field} #{self.counter}'

        def __eq__(self, other):
            return hash(self) == hash(other)

    def __init__(self, call_list, write_counter={}):
        self.call_list = call_list
        self.write_counter = write_counter
        self.reads = {}
        self.writes = {}
        self.computation_nodes = set()
        self.input_nodes = []
        self.output_nodes = []

        for c in call_list:
            if isinstance(c, KernelCall):  # TODO get rid of this one
                c = c.kernel

            if isinstance(c, KernelWrapper):
                c = c.ast

            if isinstance(c, KernelFunction):
                output_fields = c.fields_written
                input_fields = c.fields_read

                computation_node = self.ComputationNode(c)
                self.read(input_fields, computation_node)
                self.write(output_fields, computation_node)
                self.computation_nodes.add(computation_node)
            elif isinstance(c, Swap):
                computation_node = self.ComputationNode(c)
                self.read([c.field, c.destination], computation_node)
                self.write([c.field, c.destination], computation_node)
                self.computation_nodes.add(computation_node)
            elif isinstance(c, TimeloopRun):
                computation_node = ComputationGraph(c.timeloop._single_step_asts, self.write_counter)

                self.computation_nodes.add(computation_node)
            else:
                print(c)

        for c in self.computation_nodes:
            if isinstance(c, ComputationGraph):
                reads = set(c.reads.keys())
                writes = set(c.writes.keys())
                known = set(chain(self.writes.keys(), self.reads.keys()))
                c.input_nodes = [self.ArrayNode(a) for a in (known & reads)]
                c.output_nodes = [self.ArrayNode(a) for a in (known & writes)]

    def read(self, fields, kernel):
        fields = [self.FieldWriteCounter(f, self.write_counter.get(f.name, 0)) for f in fields]
        for f in fields:
            read_node = {**self.writes, **self.reads}.get(f, self.ArrayNode(f))
            read_node.destination_nodes.append(kernel)
            self.reads[f] = read_node
            kernel.input_nodes.append(read_node)

    def write(self, fields, kernel):
        for f in fields:
            field_snapshot = self.FieldWriteCounter(f, self.write_counter.get(f.name, 0) + 1)
            write_node = self.ArrayNode(field_snapshot)
            write_node.source_node = kernel
            self.writes[field_snapshot] = write_node
            kernel.output_nodes.append(write_node)
            self.write_counter[f.name] = self.write_counter.get(f.name, 0) + 1

    def to_dot(self, graph_style=None, with_code=False):
        import graphviz
        graph_style = {} if graph_style is None else graph_style

        fields = {**self.reads, **self.writes}
        dot = graphviz.Digraph(str(id(self)))

        for field, node in fields.items():
            label = f'{field.name} #{field.counter}'
            dot.node(label, style='filled', fillcolor='#a056db', label=label)

        for node in self.computation_nodes:
            if isinstance(node, ComputationGraph):
                subgraph = node.to_dot(with_code=with_code)
                dot.subgraph(subgraph)
                continue
            elif isinstance(node.kernel, Swap):
                name = f'Swap {id(node)}'
                dot.node(str(id(node)), style='filled', fillcolor='#ff5600', label=name)
            elif isinstance(node.kernel, KernelFunction):
                if with_code:
                    name = str(pystencils.show_code(node.kernel))
                else:
                    name = node.kernel.function_name

                dot.node(str(id(node)), style='filled', fillcolor='#0056db', label=name)
            else:
                raise 'foo'

            for input in node.input_nodes:
                field = input.field
                label = f'{field.name} #{field.counter}'
                dot.edge(label, str(id(node)))
            for output in node.output_nodes:
                field = output.field
                label = f'{field.name} #{field.counter}'
                dot.edge(str(id(node)), label)

        return dot

    def to_dot_file(self, path, graph_style=None, with_code=False):
        with open(path, 'w') as f:
            f.write(str(self.to_dot(graph_style, with_code)))

    class ComputationNode:
        def __init__(self, kernel):
            self.kernel = kernel
            self.input_nodes = []
            self.output_nodes = []

        def __hash__(self):
            return id(self)

    class ArrayNode:
        def __init__(self, field: Field):
            self.field = field
            self.source_node = None
            self.destination_nodes = []

        def __hash__(self):
            return id(self)
