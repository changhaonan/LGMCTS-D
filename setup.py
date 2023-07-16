from setuptools import setup

setup(
   name='lgmcts',
   version='1.0',
   description='A useful module',
   author='Haonan Chang',
   author_email='chnme40cs@gmail.com',
   packages=['lgmcts'],  #same as name
   install_requires=['numpy', 'pybullet'], #external packages as dependencies
)