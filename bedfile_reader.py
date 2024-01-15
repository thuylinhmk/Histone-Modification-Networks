import pyarrow as pa
import pyarrow.csv as pc
import pyranges as pr
import pandas as pd 
import os
import gzip
import numpy as np
#from prettytable import PrettyTable

def _read_BED_pyarrow(file_name: str, bed_type: str,):
    """Read function to pa"""
    #set up pyarrow
    block_size = 10<<20
    col_names = ['Chromosome', 'Start', 'End', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak']
    if bed_type == 'narrowPeak':
        read_options = pc.ReadOptions(
                use_threads = True, 
                column_names = col_names)
        parse_options = parse_options= pa.csv.ParseOptions(
                delimiter="\t",
                quote_char=False,
                escape_char=False,
                newlines_in_values=False)
        convert_options= pa.csv.ConvertOptions(
                column_types={
                    "Chromosome": pa.large_string(),
                    "Start": pa.int32(),
                    "End": pa.int32(),
                    "name": pa.large_string(),
                    "strand": pa.large_string(),
                    "signalValue": pa.float32(),
                    "pValue": pa.float32(),
                    "qValue": pa.float32(),
                    "peak": pa.int32()})
        data = pc.read_csv(file_name, 
                            read_options=read_options, 
                            parse_options=parse_options, 
                            convert_options=convert_options)
    else:
        raise ValueError("Not support yet!")
    return data.to_pandas()
    
    
def _read_BED_pyranges(file_name: str, bed_type: str):
    data = _read_BED_pyarrow(file_name, bed_type=bed_type)
    data_pr = pr.PyRanges(data)
    return data_pr

def _add_column_to_pr(table_pr: pr.PyRanges, col_name: str, col_data):
    table_pr.__setattr__(col_name, col_data)

class ReadBED:
    """Read BED file with pyarrow for fast read. only support unzip file, gzip, bar"""
    
    #set option for print table using pandas representation
    # display first 100 rows
    pd.set_option('display.max_rows', 100)

    # display all the  columns
    pd.set_option('display.max_columns', None)

    # set width  - 100
    pd.set_option('display.width', 100)

    # set column header -  left
    pd.set_option('display.colheader_justify', 'left')

    # set precision - 5
    pd.set_option('display.precision', 5)
    
    def __init__(
        self, 
        file_path: str, 
        bed_type: str = 'narrowPeak'):
        
        if bed_type == 'narrowPeak':
            self._data = _read_BED_pyranges(file_path, bed_type)
            self.get_peak_position()
    
    def get_data(self, to_df:bool=False):
        if to_df: 
            return self._data.as_df()
        else:
            return self._data
    
    def add_columns(self, column_name:str, column_data):
        """Add new column to data"""
        _add_column_to_pr(self._data, column_name, column_data)
           
    def get_peak_position(self, inplace=True):
        """Get position of the highest peak and add in extra column of peak position"""
        
        if 'peak_position' in self._data.columns:
            print('Peak position column is existed')
            pass
        
        if inplace:
            self._data = self._data.assign("peak_position", lambda df: df.Start + df.peak)
        else:
            df = self.get_data(to_df=True)
            df['peak_position'] = df['Start'] + df['peak']
            return df
        
    def get_overlap(self, regions: pr.PyRanges, strandedness=None, how='first', invert=False, nb_cpu=1):
        #turn region df to pyranges
        return self._data.overlap(regions, strandedness, how, invert, nb_cpu)
    
    def _create_peak_df(self):
        if 'peak_position' not in self._data.columns:
            self.get_peak_position()
    
        self.peak_pr = self._data.copy()
        self.peak_pr.Start = self._data.peak_position.to_numpy()
        self.peak_pr.End = self._data.peak_position.to_numpy()
    
    def get_peak_data(self):
        if hasattr(self, 'peak_pr') == False:
            self._create_peak_df()
        return self.peak_pr
         
    def get_peak_overlap(self, regions: pr.PyRanges, invert=False):
        if hasattr(self, 'peak_pr') == False:
            self._create_peak_df()
        
        if invert:
            return regions.overlap(self.peak_pr)
        else:
            return self.peak_pr.overlap(regions)
        