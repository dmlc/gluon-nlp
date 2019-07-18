#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Timothy Dozat
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
"""Tarjan's algorithm for strongly connected components."""

from collections import defaultdict


# ***************************************************************
class Tarjan:
    """
    Computes Tarjan's algorithm for finding strongly connected components (cycles) of a graph

    Attributes:
      edges: dictionary of edges such that edges[dep] = head
      vertices: set of dependents
      SCCs: list of sets of strongly connected components. Non-singleton sets are cycles.

    Parameters
    ----------
    prediction : numpy.ndarray
        a predicted dependency tree where prediction[dep_idx] = head_idx
    tokens : numpy.ndarray
        the tokens we care about (i.e. exclude _GO, _EOS, and _PAD)
    """
    def __init__(self, prediction, tokens):
        self._edges = defaultdict(set)
        self._vertices = set((0,))
        for dep, head in enumerate(prediction[tokens]):
            self._vertices.add(dep + 1)
            self._edges[head].add(dep + 1)
        self._indices = {}
        self._lowlinks = {}
        self._onstack = defaultdict(lambda: False)
        self._SCCs = []

        index = 0
        stack = []
        for v in self.vertices:
            if v not in self.indices:
                self.strongconnect(v, index, stack)

    # =============================================================
    def strongconnect(self, v, index, stack):
        """Find strongly connected components."""

        self._indices[v] = index
        self._lowlinks[v] = index
        index += 1
        stack.append(v)
        self._onstack[v] = True
        for w in self.edges[v]:
            if w not in self.indices:
                self.strongconnect(w, index, stack)
                self._lowlinks[v] = min(self._lowlinks[v], self._lowlinks[w])
            elif self._onstack[w]:
                self._lowlinks[v] = min(self._lowlinks[v], self._indices[w])

        if self._lowlinks[v] == self._indices[v]:
            self._SCCs.append(set())
            while stack[-1] != v:
                w = stack.pop()
                self._onstack[w] = False
                self._SCCs[-1].add(w)
            w = stack.pop()
            self._onstack[w] = False
            self._SCCs[-1].add(w)

    # ======================
    @property
    def edges(self):
        return self._edges

    @property
    def vertices(self):
        return self._vertices

    @property
    def indices(self):
        return self._indices

    @property
    def SCCs(self):
        return self._SCCs
