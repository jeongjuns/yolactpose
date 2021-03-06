# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.7
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.




"""
Module to compute DeepMatching
"""


from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_deepmatching', [dirname(__file__)])
        except ImportError:
            import _deepmatching
            return _deepmatching
        if fp is not None:
            try:
                _mod = imp.load_module('_deepmatching', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _deepmatching = swig_import_helper()
    del swig_import_helper
else:
    import _deepmatching
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0



def deepmatching_numpy(cim1, cim2, options):
    return _deepmatching.deepmatching_numpy(cim1, cim2, options)
deepmatching_numpy = _deepmatching.deepmatching_numpy

def usage_python():
    return _deepmatching.usage_python()
usage_python = _deepmatching.usage_python

from numpy import float32, rollaxis, ascontiguousarray
def deepmatching( im1=None, im2=None, options=""):
    """
    matches = deepmatching.deepmatching(image1, image2, options='')
    Compute the 'DeepMatching' between two images.
    Images must be HxWx3 numpy arrays (converted to float32).
    Options is an optional string argument ('' by default), to set the options.
    The function returns a numpy array with 6 columns, each row being x1 y1 x2 y2 score index.
     (index refers to the local maximum from which the match was retrieved)
    Version 1.2"""
    if None in (im1,im2):
      usage_python()
      return

# convert images
    if im1.dtype != float32:
        im1 = im1.astype(float32)
    if im2.dtype != float32:
        im2 = im2.astype(float32)
    assert len(im1.shape)==3 and len(im2.shape)==3, "images must have 3 dimensions"
    h, w, nchannels = im1.shape
    assert nchannels==3, "images must have 3 channels"
    im1 = ascontiguousarray(rollaxis(im1,2))
    im2 = ascontiguousarray(rollaxis(im2,2))
    corres = deepmatching_numpy( im1, im2, options)
    return corres

# This file is compatible with both classic and new-style classes.


