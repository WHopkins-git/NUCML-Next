"""
NUCML-Next Visualization Module
===============================

Comprehensive visualization tools for nuclear cross-section analysis.

Main Classes:
    CrossSectionFigure: Flexible figure class for cross-section plots
    ENDFReader: Parser for ENDF-6 formatted evaluated nuclear data files
    NNDCSigmaFetcher: Fetch processed pointwise data from NNDC Sigma

Example:
    >>> from nucml_next.visualization import CrossSectionFigure, NNDCSigmaFetcher
    >>>
    >>> # Create a figure for U-235 fission
    >>> fig = CrossSectionFigure(
    ...     isotope='U-235',
    ...     mt=18,  # Fission
    ...     title='U-235 Fission Cross Section',
    ... )
    >>>
    >>> # Add EXFOR experimental data
    >>> fig.add_exfor(exfor_df, label='EXFOR')
    >>>
    >>> # Add ENDF-B evaluated data
    >>> fig.add_endf('data/ENDF-B/neutrons/n-092_U_235.endf')
    >>>
    >>> # Add ML model predictions
    >>> fig.add_model(energies, predictions, label='XGBoost')
    >>>
    >>> # Display or save
    >>> fig.show()
    >>> fig.save('u235_fission.png')
    >>>
    >>> # For reactions with resonance structure stored in MF=2 (like Cl-35 (n,p)),
    >>> # use NNDCSigmaFetcher to get processed pointwise data:
    >>> fetcher = NNDCSigmaFetcher()
    >>> energies, xs = fetcher.get_cross_section(z=17, a=35, mt=103)
"""

from .cross_section_figure import CrossSectionFigure
from .endf_reader import ENDFReader, NNDCSigmaFetcher

__all__ = ['CrossSectionFigure', 'ENDFReader', 'NNDCSigmaFetcher']
