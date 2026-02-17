import os
import numpy as np
import json
import scipy.io as spio
import pickle

from io import BytesIO
from typing import List, Dict, Union, Set, Optional
from PIL import Image

from .itar.fold import iTarFold
from .itar.utils import stripext


class EncoderDecoder:

    '''Base EncoderDecoder class.

    This is an abstract class intended to be subclassed for building encoders
    and decoders for different data modalities. It provides a framework for
    encoding and decoding operations, supporting various file extensions.

    To implement a subclass, it must include:
    - `_encode` and `_decode` methods to handle the specific logic of encoding
      and decoding.
    - A `supported_extensions` attribute, which specifies the file extensions
      that the encoder/decoder supports.

    Encoders or decoders can be instantiated using the class methods `encoder`
    or `decoder`, respectively. Alternatively, they can be initialized directly
    with the desired mode ('encode' or 'decode').

    Parameters
    ----------
    mode : str
        The mode of operation. It must be either 'encode' or 'decode'.

    Attributes
    ----------
    supported_extensions : Optional[Union[List, Dict, Set]]
        A collection of file extensions that the encoder/decoder supports.
    _mode : str
        The mode of operation, either 'encode' or 'decode'.
    _bit8 : int
        An integer representing the 8-bit limit (2^8).
    _bit16 : int
        An integer representing the 16-bit limit (2^16).
    _bit24 : int
        An integer representing the 24-bit limit (2^24).

    Methods
    -------
    supports(filepath)
        Returns True if the provided file path is supported by the encoder/decoder.
    encoder()
        Class method to initialize an encoder instance.
    decoder()
        Class method to initialize a decoder instance.
    __call__(x)
        Encodes or decodes the provided data based on the mode of the instance.

    Raises
    ------
    NotImplementedError
        If `_encode` or `_decode` methods are not implemented in a subclass.
    ValueError
        If an invalid mode is provided.

    Examples
    --------
    >>> class MyEncoder(EncoderDecoder):
    ...     # Implementation of `_encode` and `_decode` methods,
    ...     # and `supported_extensions` attribute.
    ...
    >>> encoder = MyEncoder.encoder()
    >>> decoder = MyEncoder.decoder()
    '''
    def __init__(self, mode:str, default_to_identity:bool=False):
        '''Initializes the encoder / decoder. 
        
        Parameters
        ----------
        mode : str
            The mode to apply, either `encode` or `decode`.
        default_to_identity : bool, optional
            Whether the extension can safely default to identity maps for transforms.
        '''
        self.supported_extensions:Optional[Union[List,Dict,Set]] = None
        assert mode in ['encode', 'decode']
        self._mode = mode
        self._bit8 = 2**8
        self._bit16 = 2**16
        self._bit24 = 2**24
        self._default_tv = None
        self._default_to_identity = default_to_identity

    def supports(self, filepath:str):
        '''Returns True if the provided path is supported.

        NOTE: This does not check if the file exists.

        Parameters
        ----------
        filepath : str 
            A file path. 
        
        Returns
        -------
        bool
            Whether the file extension is supported by the encoder/decoder.
        '''
        assert self.supported_extensions is not None
        name, extension = os.path.splitext(filepath)
        return extension in self.supported_extensions

    def _encode(self, obj):
        raise NotImplementedError('EncodeDecoder is abstract!')
    
    def _decode(self, data:bytes):
        raise NotImplementedError('EncoderDecoder is abstract!')
    
    def __call__(self, x):
        if self._mode == 'encode':
            return self._encode(x)
        elif self._mode == 'decode':
            return self._decode(x)
        else:
            raise ValueError(f'Invalid mode: {self._mode}')
        
    @classmethod
    def encoder(cls):
        '''Initializes an encoder.
        '''
        return cls('encode')

    @classmethod
    def decoder(cls):
        '''Initializes a decoder.
        '''
        return cls('decode')


class PIL(EncoderDecoder):

    '''PIL Image EncoderDecoder class.

    This subclass of EncoderDecoder is a default image encoder and decoder
    for images, utilizing the Python Imaging Library (PIL). It is specialized
    in handling various image formats.

    Note
    ----
    Image extensions are generally considered raw image data. For masks 
    and other alternate image modalities, it is recommended to use either the 
    provided segmentation encoders or a custom implementation.

    Parameters
    ----------
    mode : str
        The mode of operation, either 'encode' for encoding images to bytes, or
        'decode' for decoding bytes to images.

    Attributes
    ----------
    supported_extensions : dict
        A dictionary of file extensions supported by PIL for encoding and decoding.

    Methods
    -------
    _decode(data)
        Decodes byte data into a PIL Image object.
    _encode(img)
        Encodes a PIL Image object into byte data.

    Raises
    ------
    AssertionError
        If the image to be encoded does not have a set format.

    Examples
    --------
    >>> encoder = PIL.encoder()
    >>> decoder = PIL.decoder()
    >>> image = decoder(some_byte_data)
    >>> byte_data = encoder(image)
    '''
    def __init__(self, mode:str, convert_to:Optional[str]='RGB'):
        '''Initializes the class label encoder/decoder.

        Parameters
        ----------
        mode : str
            The mode of operation, either 'encode' or 'decode'.

        Raises
        ------
        ValueError
            If an invalid mode is specified.
        '''
        super().__init__(mode)
        self.supported_extensions = Image.registered_extensions()
        self.convert_to = convert_to
        self._default_tv = 'image'

    def _decode(self, data:bytes):
        img = Image.open(BytesIO(data))
        if self.convert_to is not None:
            return img.convert(self.convert_to)
        return img
    
    def _encode(self, obj:Image.Image):
        assert obj.format is not None, f'PIL image must have a set format!'
        byte_arr = BytesIO()
        obj.save(byte_arr, format=obj.format)
        return byte_arr.getvalue()


class CLS(EncoderDecoder):

    '''Class label EncoderDecoder.

    This subclass of EncoderDecoder is specialized for encoding and decoding
    class labels for single-label classification tasks. It handles the conversion
    between integer labels and their byte representation.

    Parameters
    ----------
    mode : str
        The mode of operation, either 'encode' for encoding integer labels to bytes,
        or 'decode' for decoding byte data to integer labels.

    Attributes
    ----------
    supported_extensions : dict
        A dictionary mapping the file extension '.cls' to a description, indicating
        the support for classification label data.

    Examples
    --------
    >>> encoder = CLS.encoder()
    >>> decoder = CLS.decoder()
    >>> encoded_label = encoder(5)  # Encoding the label '5'
    >>> decoded_label = decoder(encoded_label)  # Decoding back to integer label
    '''
    def __init__(self, mode:str):
        '''Initializes the encoder / decoder. 
        
        Args:
            mode (str): The mode, either `encode` or `decode`.
        '''
        super().__init__(mode, True)
        self.supported_extensions = {'.cls': 'Classification label'}
        self._default_tv = 'other'

    def _decode(self, data:bytes):
        return int(data)
    
    def _encode(self, obj:int):
        return f'{obj}'.encode()
    

class PseudoExtension(EncoderDecoder):

    '''PseudoExtension EncoderDecoder.

    This subclass of EncoderDecoder handles pseudo labels. 
    It handles possible pseudo-extensions like filenames and indices.

    Parameters
    ----------
    mode : str
        The mode of operation, either 'encode' for encoding integer labels to bytes,
        or 'decode' for decoding byte data to integer labels.
    '''

    # Code for pseudo extensions, to enable returning index or filenames
    _valid_pseudo_extensions = tuple(iTarFold._pseudoextensions)

    def __init__(self, mode:str):
        super().__init__(mode, True)
        self.supported_extensions = {
            '_idx': 'Index of file in filtered dataset.',
            '_name': 'File / stem name.', 
            '_stem': 'File / stem name.',
            '_fid': 'Fold / file ID, (e.g. <fold>_<fid>.tar).'
        }
        if mode == 'encode':
            raise ValueError(f'PseudoExtensions do not support encoding mode!')
        self._default_tv = 'other'



class RLE(EncoderDecoder):

    '''Simple run-length EncoderDecoder.

    This subclass of EncoderDecoder implements a simple run-length encoding (RLE),
    typically used for largely homogeneous data such as segmentation masks. The RLE
    implementation supports arbitrary dimensions, making it suitable for higher-dimensional
    segmentation tasks beyond 2D image data.

    Notes
    -----
    - This RLE is not the same as the RLE encoding used in the COCO API.
    - The encoder accepts numpy ndarrays of arbitrary shape as input. The output file
      consists of a header and the RLE data. The header contains a single byte encoding
      the from/to dtypes for compression, the shape and table lengths in bytes, the shape
      of the array, and the lookup table for the array values.
    - While RLE encoding typically requires more space (1.5x-6x) than PNG encoded
      segmentation, it supports up to 64-bit resolution masks and is significantly faster
      (up to 8x-15x faster) than PNG or QOI formats for encoding/decoding segmentation masks.
      This makes RLE competitive for large batch sizes despite the increased space requirement.

    Parameters
    ----------
    mode : str
        The mode of operation, either 'encode' or 'decode'.

    Attributes
    ----------
    supported_extensions : dict
        A dictionary mapping the '.rle' file extension to a description, indicating
        support for run-length encoding data.

    Methods
    -------
    pack_dtypes(olddt, newdt)
        Packs the original and new data types into a single byte.
    unpack_dtypes(dtbyte)
        Unpacks the original and new data types from a single byte.
    unique_consecutive_with_counts(arr)
        Finds unique consecutive elements and their counts for RLE encoding.

    Examples
    --------
    >>> encoder = RLE.encoder()
    >>> decoder = RLE.decoder()
    >>> encoded_data = encoder(some_ndarray)
    >>> decoded_data = decoder(encoded_data)
    '''

    def __init__(self, mode:str):
        '''Initializes the run-length encoder/decoder.

        Parameters
        ----------
        mode : str
            The mode of operation, either 'encode' or 'decode'.

        Raises
        ------
        ValueError
            If an invalid mode is specified.
        '''
        super().__init__(mode)
        self.supported_extensions = {
            '.rle': 'Run length encoding',
        }
        self._default_tv = 'mask'

    @staticmethod
    def pack_dtypes(olddt:np.dtype, newdt:np.dtype):
        '''Packs original and new data types into a single byte for RLE header.

        Parameters
        ----------
        olddt : np.dtype
            The original data type of the array. Can be either signed or unsigned.
        newdt : np.dtype
            The new data type, inferred for compression. Always unsigned.

        Returns
        -------
        bytes
            A single byte encoding the old and new data types.
        '''
        old = int(np.log2(np.iinfo(olddt).bits)) - 3
        new = int(np.log2(np.iinfo(newdt).bits)) - 3
        sign = 1 if np.issubdtype(olddt, np.signedinteger) else 0
        old = (old << 1) | sign
        return (old << 4 | new).to_bytes(1, 'big')
    
    @staticmethod
    def unpack_dtypes(dtbyte:bytes):
        '''Unpacks original and new data types from a single byte used in RLE header.

        Parameters
        ----------
        dtbyte : bytes
            A byte object encoding the old and new data types.

        Returns
        -------
        Tuple[np.dtype, np.dtype]
            The original and new data types, respectively.
        '''
        dtint = int.from_bytes(dtbyte, 'big')
        old, new = dtint >> 4, dtint & 15
        sign = old & 1
        old = (old >> 1)
        oldbit = 2**(old+3)
        newbit = 2**(new+3)
        olddt = np.dtype(('int' if sign else 'uint') + str(oldbit))
        newdt = np.dtype('uint' + str(newbit))
        return olddt, newdt

    @staticmethod    
    def unique_consecutive_with_counts(arr):
        '''Finds unique consecutive elements and their counts in an array for RLE encoding.

        Parameters
        ----------
        arr : np.ndarray
            A flattened array to encode with RLE.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing arrays of unique consecutive values and their counts.
        '''
        assert arr.ndim == 1
        mask = np.empty(arr.shape, dtype=np.bool_)
        mask[0] = True
        mask[1:] = arr[1:] != arr[:-1]
        unique_consecutive = arr[mask]
        counts = np.diff(np.append(np.where(mask), len(arr)))
        return unique_consecutive, counts    

    def _decode(self, data:bytes):
        # Unpack dtypes
        olddt, newdt = self.unpack_dtypes(data[0:1])
        
        # Unpack lengths and get endpoints
        shapelen = int.from_bytes(data[1:9], 'big')
        tablelen = int.from_bytes(data[9:17], 'big')
        shape_end = 17+shapelen
        table_end = shape_end + tablelen
        
        # Get shape, table and RLE bytes
        shapebytes = data[17:shape_end]
        tablebytes = data[shape_end:table_end]
        rlebytes = data[table_end:]
        
        # Get indices and counts
        unq, cnt = np.frombuffer(rlebytes, dtype=newdt).reshape(2, -1)
        table = np.frombuffer(tablebytes, dtype=olddt)
        
        # Reconstruct array
        vals = table[unq]
        shape = np.frombuffer(shapebytes, dtype=np.uint64)

        return np.repeat(vals, cnt).reshape(*shape)
    
    def _encode(self, obj:np.ndarray):
        # Get dtype, shape, and size info
        olddt = obj.dtype
        shape = obj.shape
        
        # Compute RLE
        unq, cnt = self.unique_consecutive_with_counts(obj.flatten())
        table = np.unique(unq)
        maxunq = len(table)
        
        # Remap and determine new dtype
        mapunq = np.searchsorted(table, unq)
        newdt = np.min_scalar_type(max(maxunq, cnt.max()))
            
        # Construct rlebytes
        rle = np.stack((mapunq, cnt), axis=0)
        rlebytes = rle.astype(newdt).tobytes()
        
        # Construct header
        dtbyte = self.pack_dtypes(olddt, newdt)
            
        # Construct shape and table
        shapebytes = np.array(shape, dtype=np.uint64).tobytes()
        tablebytes = table.tobytes()
        
        # Determine lengths
        shapelen = len(shapebytes).to_bytes(8, 'big')
        tablelen = len(tablebytes).to_bytes(8, 'big')
        
        # Concatenate
        return dtbyte + shapelen + tablelen + shapebytes + tablebytes + rlebytes
    

class SEG8(EncoderDecoder):

    '''8-bit segmentation mask EncoderDecoder.

    This subclass of EncoderDecoder is tailored for encoding and decoding 8-bit
    segmentation masks, with a maximum of 256 classes. It utilizes PNG encoding,
    which provides efficient compression for this type of data.

    Notes
    -----
    - PNG is chosen over QOI for segmentation masks due to its superior performance
      in terms of compression effectiveness for this specific application.
      Although QOI offers similar performance to RLE, the Python interface seems
      to indicate that it is considerably slower compared to PNG for encoding 
      segmentation masks.

    Parameters
    ----------
    mode : str
        The mode of operation, either 'encode' or 'decode'.

    Attributes
    ----------
    supported_extensions : dict
        A dictionary mapping the '.seg8' file extension to a description, indicating
        support for 8-bit segmentation masks encoded in PNG format.

    Examples
    --------
    >>> encoder = SEG8.encoder()
    >>> decoder = SEG8.decoder()
    >>> encoded_mask = encoder(some_ndarray)
    >>> decoded_mask = decoder(encoded_mask)
    '''

    def __init__(self, mode:str):
        '''Initializes the 8-bit segmentation mask encoder/decoder.

        Parameters
        ----------
        mode : str
            The mode of operation, either 'encode' or 'decode'.

        Raises
        ------
        ValueError
            If an invalid mode is specified.
        '''
        super().__init__(mode)
        self.supported_extensions = {
            '.seg8': '8 bit segmentation mask (PNG)',
        }
        self._default_tv = 'mask'
    
    def _encode(self, obj:np.ndarray):
        assert obj.dtype == np.uint8
        img = Image.fromarray(obj, 'L')
        byte_arr = BytesIO()
        img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()
    
    def _decode(self, data:bytes):
        byte_arr = BytesIO(data)
        img = Image.open(byte_arr).convert('L')
        return np.array(img)


class SEG16(EncoderDecoder):

    """16-bit segmentation mask EncoderDecoder.

    This subclass of EncoderDecoder is designed for encoding and decoding 16-bit
    segmentation masks, allowing for a maximum of 65536 classes. It uses PNG encoding
    for efficient compression and decompression.

    Notes
    -----
    - The choice of PNG over QOI for segmentation masks is due to PNG's superior
      performance in this specific application.

    Parameters
    ----------
    mode : str
        The mode of operation, either 'encode' or 'decode'.

    Attributes
    ----------
    supported_extensions : dict
        A dictionary mapping the '.seg16' file extension to a description, indicating
        support for 16-bit segmentation masks encoded in PNG format.

    Examples
    --------
    >>> encoder = SEG16.encoder()
    >>> decoder = SEG16.decoder()
    >>> encoded_mask = encoder(some_ndarray)
    >>> decoded_mask = decoder(encoded_mask)
    """

    def __init__(self, mode:str):
        """Initializes the 16-bit segmentation mask encoder/decoder.

        Parameters
        ----------
        mode : str
            The mode of operation, either 'encode' or 'decode'.
        
        Raises
        ------
        ValueError
            If an invalid mode is specified.
        """        
        super().__init__(mode)
        self.supported_extensions = {
            '.seg16': '16 bit segmentation mask (PNG)',
        }
        self._default_tv = 'mask'
    
    def _encode(self, obj:np.ndarray):
        assert obj.dtype == np.uint16
        img = Image.fromarray(obj, 'I;16')
        byte_arr = BytesIO()
        img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()
    
    def _decode(self, data:bytes):
        byte_arr = BytesIO(data)
        img = Image.open(byte_arr).convert('I;16')
        return np.array(img)
    

class SEG24(EncoderDecoder):

    """24-bit segmentation mask EncoderDecoder.

    This subclass of EncoderDecoder is tailored for encoding and decoding 24-bit
    segmentation masks, capable of handling up to 16777216 classes. It employs PNG
    encoding, which offers efficient compression for this type of data.

    Notes
    -----
    - PNG is used for its better performance compared to QOI in encoding segmentation masks.

    Parameters
    ----------
    mode : str
        The mode of operation, either 'encode' or 'decode'.

    Attributes
    ----------
    supported_extensions : dict
        A dictionary mapping the '.seg24' file extension to a description, indicating
        support for 24-bit segmentation masks encoded in PNG format.

    Examples
    --------
    >>> encoder = SEG24.encoder()
    >>> decoder = SEG24.decoder()
    >>> encoded_mask = encoder(some_ndarray)
    >>> decoded_mask = decoder(encoded_mask)
    """
    def __init__(self, mode:str):
        """Initializes the 24-bit segmentation mask encoder/decoder.

        Parameters
        ----------
        mode : str
            The mode of operation, either 'encode' or 'decode'.
        
        Raises
        ------
        ValueError
            If an invalid mode is specified.
        """        
        super().__init__(mode)
        self.supported_extensions = {
            '.seg24': '24 bit segmentation mask (PNG|RGB)',
        }
        self._default_tv = 'mask'

    def _torgb(self, arr:np.ndarray) -> np.ndarray:
        b = (arr % self._bit8).astype(np.uint8)
        g = ((arr // self._bit8) % self._bit8).astype(np.uint8)
        r = (arr // self._bit16).astype(np.uint8)
        return np.stack((r,g,b), 2)

    def _fromrgb(self, arr:np.ndarray) -> np.ndarray:
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        return r * self._bit16 + g * self._bit8 + b

    def _encode(self, obj:np.ndarray):
        assert obj.dtype == np.uint32
        img = Image.fromarray(self._torgb(obj), 'RGB')
        byte_arr = BytesIO()
        img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()
    
    def _decode(self, data:bytes):
        byte_arr = BytesIO(data)
        img = Image.open(byte_arr).convert('RGB')
        return self._fromrgb(np.array(img))


class SEG32(EncoderDecoder):

    """32-bit segmentation mask EncoderDecoder.

    This subclass of EncoderDecoder is designed for encoding and decoding 32-bit
    segmentation masks, with a theoretical maximum of 4294967296 classes. It utilizes
    PNG encoding for efficient data handling.

    Notes
    -----
    - Although capable of high resolution, such level of detail is typically unnecessary
      for most segmentation tasks.
    - PNG is preferred over QOI due to its better performance in encoding segmentation masks.

    Parameters
    ----------
    mode : str
        The mode of operation, either 'encode' or 'decode'.

    Attributes
    ----------
    supported_extensions : dict
        A dictionary mapping the '.seg32' file extension to a description, indicating
        support for 32-bit segmentation masks encoded in PNG format.

    Examples
    --------
    >>> encoder = SEG32.encoder()
    >>> decoder = SEG32.decoder()
    >>> encoded_mask = encoder(some_ndarray)
    >>> decoded_mask = decoder(encoded_mask)
    """
    def __init__(self, mode:str):
        """Initializes the 32-bit segmentation mask encoder/decoder.

        Parameters
        ----------
        mode : str
            The mode of operation, either 'encode' or 'decode'.
        
        Raises
        ------
        ValueError
            If an invalid mode is specified.
        """        
        super().__init__(mode)
        self.supported_extensions = {
            '.seg32': '32 bit segmentation mask (PNG|RGBA)',
        }
        self._default_tv = 'mask'

    def _torgba(self, arr:np.ndarray) -> np.ndarray:
        a = (arr % self._bit8).astype(np.uint8)
        b = ((arr // self._bit8) % self._bit8).astype(np.uint8)
        g = ((arr // self._bit16) % self._bit16).astype(np.uint8)
        r = (arr // self._bit24).astype(np.uint8)
        return np.stack((r,g,b,a), 2)

    def _fromrgba(self, arr:np.ndarray) -> np.ndarray:
        r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
        return r * self._bit24 + g * self._bit16 + b * self._bit8 + a

    def _encode(self, obj:np.ndarray):
        assert obj.dtype == np.uint32
        img = Image.fromarray(self._torgba(obj), 'RGBA')
        byte_arr = BytesIO()
        img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()
    
    def _decode(self, data:bytes):
        byte_arr = BytesIO(data)
        img = Image.open(byte_arr).convert('RGBA')
        return self._fromrgba(np.array(img))
    

class JSON(EncoderDecoder):
    """JSON encoding / decoding for text, trees, string objects, etc.

    This subclass of EncoderDecoder specializes in encoding and decoding data structures
    such as lists and dictionaries into/from JSON format using UTF-8 encoding. It is
    designed to handle text-based data representations commonly used for data interchange.

    Note
    ----
    - This implementation uses UTF-8 encoding. For other encodings, a custom
      EncoderDecoder should be implemented.

    Parameters
    ----------
    mode : str
        The mode of operation, either 'encode' or 'decode'.

    Attributes
    ----------
    supported_extensions : dict
        A dictionary mapping the '.json' file extension to a description, indicating
        support for JSON formatted files.

    Examples
    --------
    >>> encoder = JSON.encoder()
    >>> decoder = JSON.decoder()
    >>> encoded_data = encoder(some_dict_or_list)
    >>> decoded_data = decoder(encoded_data)
    """
    def __init__(self, mode:str):
        """Initializes the JSON encoder/decoder.

        Parameters
        ----------
        mode : str
            The mode of operation, either 'encode' or 'decode'.

        Raises
        ------
        ValueError
            If an invalid mode is specified.
        """        
        super().__init__(mode)
        self.supported_extensions = {
            '.json': 'JSON file.',
        }
        self._default_tv = 'other'

    def _encode(self, obj:Union[List, Dict]):
        return json.dumps(obj).encode()

    def _decode(self, data:bytes):
        return json.loads(data)


class NPY(EncoderDecoder):
    
    """Numpy array encoding/decoding.

    This subclass of EncoderDecoder is specifically designed for encoding and decoding
    numpy arrays. It uses the standard numpy `save` and `load` functions to handle
    `.npy` files, which are binary files for storing numpy arrays.

    Note
    ----
    - This implementation uses numpy's `save` and `load` methods without any compression.
      For compression options, a custom implementation or a different file format should be considered.

    Parameters
    ----------
    mode : str
        The mode of operation, either 'encode' or 'decode'.

    Attributes
    ----------
    supported_extensions : dict
        A dictionary mapping the '.npy' file extension to a description, indicating
        support for numpy array files.

    Examples
    --------
    >>> encoder = NPY.encoder()
    >>> decoder = NPY.decoder()
    >>> encoded_array = encoder(some_numpy_array)
    >>> decoded_array = decoder(encoded_array)
    """
    def __init__(self, mode:str):
        """Initializes the numpy array encoder/decoder.

        Parameters
        ----------
        mode : str
            The mode of operation, either 'encode' or 'decode'.
        
        Raises
        ------
        ValueError
            If an invalid mode is specified.
        """        
        super().__init__(mode)
        self.supported_extensions = {
            '.npy': 'Numpy array file.',
        }
        self._default_tv = 'other'

    def _encode(self, obj:np.ndarray):
        byte_arr = BytesIO()
        np.save(byte_arr, obj)
        return byte_arr.getvalue()
    
    def _decode(self, data:bytes):
        byte_arr = BytesIO(data)
        return np.load(byte_arr)


class MAT(EncoderDecoder):

    """Matlab file array encoding/decoding.

    This subclass of EncoderDecoder is specifically designed for encoding and decoding
    MATLAB arrays into and from `.mat` files. It uses scipy's `savemat` and `loadmat`
    functions for handling `.mat` files, providing an interface for MATLAB data in Python.

    Note
    ----
    - This implementation supports MATLAB version 5 and above. For MATLAB versions 4 and below,
      a custom EncoderDecoder would need to be implemented due to different file format specifications.

    Parameters
    ----------
    mode : str
        The mode of operation, either 'encode' or 'decode'.

    Attributes
    ----------
    supported_extensions : dict
        A dictionary mapping the '.mat' file extension to a description, indicating
        support for MATLAB file arrays.

    Examples
    --------
    >>> encoder = MAT.encoder()
    >>> decoder = MAT.decoder()
    >>> encoded_mat = encoder(some_dict_representing_matlab_data)
    >>> decoded_mat = decoder(encoded_mat)
    """
    def __init__(self, mode:str):
        """Initializes the MATLAB file array encoder/decoder.

        Parameters
        ----------
        mode : str
            The mode of operation, either 'encode' or 'decode'.
        
        Raises
        ------
        ValueError
            If an invalid mode is specified.
        """        
        super().__init__(mode)
        self.supported_extensions = {
            '.mat': 'Matlab file.',
        }
        self._default_tv = 'other'
    
    def _encode(self, obj:dict):
        byte_arr = BytesIO()
        spio.savemat(byte_arr, obj)
        return byte_arr.getvalue()
    
    def _decode(self, data:bytes):
        byte_arr = BytesIO(data)
        return spio.loadmat(byte_arr)
    

class PKL(EncoderDecoder):

    """Pickle encoding/decoding for Python objects.

    This subclass of EncoderDecoder is designed for serializing and deserializing Python
    objects using pickle. It is suitable for a wide range of Python objects, enabling
    their storage and retrieval in a binary format.

    Note
    ----
    - This implementation uses pickle protocol 5. For other versions of the pickle protocol,
      a custom EncoderDecoder would need to be implemented. Protocol 5 was chosen for its
      efficiency and support for out-of-band data.

    Parameters
    ----------
    mode : str
        The mode of operation, either 'encode' or 'decode'.

    Attributes
    ----------
    supported_extensions : dict
        A dictionary mapping the '.pkl' file extension to a description, indicating
        support for Python pickled object files.

    Examples
    --------
    >>> encoder = PKL.encoder()
    >>> decoder = PKL.decoder()
    >>> encoded_obj = encoder(some_python_object)
    >>> decoded_obj = decoder(encoded_obj)
    """
    def __init__(self, mode:str):
        """Initializes the pickle encoder/decoder.

        Parameters
        ----------
        mode : str
            The mode of operation, either 'encode' or 'decode'.
        
        Raises
        ------
        ValueError
            If an invalid mode is specified.
        """        
        super().__init__(mode)
        self.supported_extensions = {
            '.pkl': 'Python pickled object file.',
        }
        self._default_tv = 'other'
    
    def _encode(self, obj):
        return pickle.dumps(obj)
    
    def _decode(self, data:bytes):
        return pickle.loads(data)


class TXT(EncoderDecoder):
    """Plain text encoder/decoder for `.txt` files.

    This subclass of EncoderDecoder is designed for handling UTF-8 encoded text files.
    It supports reading and writing string data as bytes.

    Parameters
    ----------
    mode : str
        The mode of operation, either 'encode' or 'decode'.

    Attributes
    ----------
    supported_extensions : dict
        A dictionary mapping the '.txt' file extension to a description, indicating
        support for UTF-8 text files.

    Examples
    --------
    >>> encoder = TXT.encoder()
    >>> decoder = TXT.decoder()
    >>> encoded = encoder("Hello, world!")
    >>> decoded = decoder(encoded)  # "Hello, world!"
    """
    def __init__(self, mode: str):
        super().__init__(mode, True)
        self.supported_extensions = {
            '.txt': 'Plain UTF-8 encoded text file.',
        }
        self._default_tv = 'other'

    def _encode(self, obj: str):
        assert isinstance(obj, str)
        return obj.encode('utf-8')

    def _decode(self, data: bytes):
        return data.decode('utf-8')


# Define default encoders

_default_decoder_list = [
    PIL.decoder(),
    CLS.decoder(),
    PseudoExtension.decoder(),
    RLE.decoder(),
    SEG8.decoder(),
    SEG16.decoder(),
    SEG24.decoder(),
    SEG32.decoder(),
    JSON.decoder(),
    NPY.decoder(),
    MAT.decoder(),
    PKL.decoder(),
    TXT.decoder(),
]

_default_encoder_list = [
    PIL.encoder(),
    CLS.encoder(),
    RLE.encoder(),
    SEG8.encoder(),
    SEG16.encoder(),
    SEG24.encoder(),
    SEG32.encoder(),
    JSON.encoder(),
    NPY.encoder(),
    MAT.encoder(),
    PKL.encoder(),
    TXT.encoder(),
]

DEFAULT_DECODERS = {
    stripext(ext):dec 
    for dec in _default_decoder_list 
    for ext in dec.supported_extensions    
}

DEFAULT_ENCODERS = {
    stripext(ext):dec 
    for dec in _default_encoder_list 
    for ext in dec.supported_extensions    
}
