from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from libcpp.pair cimport pair
cimport numpy as np
import cython

cdef extern from "Log_Uniform_Sampler.h":
    cdef cppclass Log_Uniform_Sampler:
        Log_Uniform_Sampler(int) except +
        unordered_set[long] sample_unique(int, int*) except +

cdef class LogUniformSampler:
    cdef Log_Uniform_Sampler* c_sampler

    def __cinit__(self, N):
        self.c_sampler = new Log_Uniform_Sampler(N)

    def __dealloc__(self):
        del self.c_sampler

    def sample_unique(self, size):
        cdef int num_tries
        samples = list(self.c_sampler.sample_unique(size, &num_tries))
        return samples, num_tries
