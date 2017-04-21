from setuptools import setup

setup(name='ranknn',
      version='0.1',
      description='Learning to Rank with Keras and Tensorflow',
      url='',
      author='ZZ',
      author_email='',
      license='MIT',
      packages=['ranknn'],
      install_requires=[
      	'tensorflow>=1',
      	'keras>=2',
      	'tqdm',
        'numpy>=1.12',
        'scikit-learn>=0.18'
      ],
      zip_safe=False)
