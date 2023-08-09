# credits to: https://e-dorigatti.github.io/python/2020/06/19/multiprocessing-large-objects.html

from multiprocessing.shared_memory import SharedMemory
import numpy as np


class SharedNumpyArray:
    '''
    Wraps a numpy array so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing.
    '''

    def __init__(self, array):
        '''
        Creates the shared memory and copies the array therein
        '''
        # create the shared memory location of the same size of the array
        self._shared = SharedMemory(create=True, size=array.nbytes)

        # save data type and shape, necessary to read the data correctly
        self._dtype, self._shape = array.dtype, array.shape

        # create a new numpy array that uses the shared memory we created.
        # at first, it is filled with zeros
        res = np.ndarray(
            self._shape, dtype=self._dtype, buffer=self._shared.buf
        )

        # copy data from the array to the shared memory. numpy will
        # take care of copying everything in the correct format
        res[:] = array[:]

    def read(self):
        '''
        Reads the array from the shared memory without unnecessary copying.
        '''
        # simply create an array of the correct shape and type,
        # using the shared memory location we created earlier
        return np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)

    def copy(self):
        '''
        Returns a new copy of the array stored in shared memory.
        '''
        return np.copy(self.read_array())

    def unlink(self):
        '''
        Releases the allocated memory. Call when finished using the data,
        or when the data was copied somewhere else.
        '''
        self._shared.close()
        self._shared.unlink()
