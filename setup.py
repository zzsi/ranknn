from setuptools import setup

setup(name='rankpy',
      version='0.1',
      description='Learning to Rank',
      url='',
      author='ZZ',
      author_email='',
      license='MIT',
      packages=['rankpy'],
      install_requires=[
      	'tensorflow>=1',
      	'keras>=2',
      	'tqdm',
        'numpy>=1.12',
        'scikit-learn>=0.18'
      ],
      zip_safe=False)
