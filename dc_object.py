from dataclasses import dataclass, field
from typing import List

@dataclass
class DataCraterAnalysis:
    import pandas as pd
    name: str = field(default='',
                      metadata={'description':'Name of the original STL file'})
    points: pd.DataFrame = field(default=pd.DataFrame(),
                                 metadata={'description':'Mesh points. pd.DataFrame. Columns: [x,y,z]'},
                                 compare=False,
                                 repr=False)
    # anderer Datentyp, damit man mehrere slices drin speichern kann
    cross_section: list() = field(default_factory=list,
                                 compare=False,
                                 repr=False)
    fit: pd.DataFrame = field(default=pd.DataFrame(),
                               metadata={'description':'Parameters of the fitted sphere. pd.DataFrame. Columns: [center_x, center_y, center_z, radius, error]'},
                               compare=False,
                               repr=False)
    # mesh einfügen?
    def __repr__(self):
        number_of_slices = len(self.cross_section)
        name = self.name
        return  f'Name: {name}, Number of slices: {number_of_slices}'

