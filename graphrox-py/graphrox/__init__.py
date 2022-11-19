import ctypes
import os
import platform


_c_csv_libs_path = os.path.join(Path(_file_).resolve().parent, 'libs')

_os_name = platform.system()
# https://en.wikipedia.org/wiki/Uname
if _os_name == 'Linux':
    if 'x86_64' == platform.uname().machine:
        _dll_name = 'csv-to-json-x86.dylib'
    elif 'arm' in platform.uname().machine.lower() or 'aarch64' in platform.uname().machine.lower():
        _dll_name = 'csv-to-json-arm64.dylib'
    else:
        raise ImportError('csvtojson module not supported on this system')
elif _os_name == 'Windows':
    if 'x86_64' == platform.uname().machine:
        _dll_name = 'csv-to-json-x86.dylib'
    elif 'arm' in platform.uname().machine.lower() or 'aarch64' in platform.uname().machine.lower():
        _dll_name = 'csv-to-json-arm64.dylib'
    else:
        raise ImportError('csvtojson module not supported on this system')
elif _os_name == 'Darwin':
    if 'x86_64' == platform.uname().machine:
        _dll_name = 'csv-to-json-x86.dylib'
    elif 'arm' in platform.uname().machine.lower() or 'aarch64' in platform.uname().machine.lower():
        _dll_name = 'csv-to-json-arm64.dylib'
    else:
        raise ImportError('csvtojson module not supported on this system')
elif _os_name == 'FreeBSD':
    if 'x86_64' == platform.uname().machine:
        _dll_name = 'csv-to-json-x86.dylib'
    elif 'arm' in platform.uname().machine.lower() or 'aarch64' in platform.uname().machine.lower():
        _dll_name = 'csv-to-json-arm64.dylib'
    else:
        raise ImportError('csvtojson module not supported on this system')
else:
    raise ImportError('csvtojson module not supported on this system')
