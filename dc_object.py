from dataclasses import dataclass, field


@dataclass
class DataCraterAnalysis:
    import pandas as pd

    name: str = field(
        default="", metadata={"description": "Name of the original STL file"}
    )
    points: pd.DataFrame = field(
        default=pd.DataFrame(),
        metadata={"description": "Mesh points. pd.DataFrame. Columns: [x,y,z]"},
        compare=False,
        repr=False,
    )
    cross_section: pd.DataFrame = field(
        default=pd.DataFrame(), compare=False, repr=False
    )
    fit: pd.DataFrame = field(
        default=pd.DataFrame(),
        metadata={
            "description": "Parameters of the fitted sphere. pd.DataFrame. Columns: [center_x, center_y, center_z, radius, error]"
        },
        compare=False,
        repr=False,
    )
    # mesh einfügen?
    def __repr__(self):
        if hasattr(self.cross_section.columns, "levels"):
            slices = self.cross_section.columns.levels[0].to_list()
        else:
            slices = "None"
        name = self.name
        return f"Name: {name}, {len(slices)} Slices: {slices}"
