from setuptools import setup, Distribution
 
 
class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True


setup(
    name='graphrox',
    version='1.0.2',
    author="Tanner Davies",
    description='A network graph library for efficiently compressing and generating approximations of graphs',
    packages=['graphrox'],
    package_data={
        'graphrox': ['libeay32.dll'],
    },
    distclass=BinaryDistribution
)
