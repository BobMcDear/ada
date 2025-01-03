from setuptools import setup, find_packages


setup(
    name='ada',
    version='0.0.1',
    description='Source-to-source autodiff for APL.',
    long_description=open('README.md').read(),
	long_description_content_type='text/markdown',
    author='Borna Ahmadzadeh',
	author_email='borna.ahz@gmail.com',
	url='https://github.com/bobmcdear/ada',
    packages=find_packages(),
    license='MIT',
    keywords=['ad', 'apl', 'autodiff', 'sct', 'tangent'],
    entry_points={'console_scripts': ['ada = src.cli:main']},
    install_requires=['astor>=0.8.0', 'tangent @ git+https://github.com/smbarr/tangent-upgrade-gast.git'],
    python_requires='>=3.8',
)
